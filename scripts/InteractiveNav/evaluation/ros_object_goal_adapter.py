"""Evaluator-owned ROS adapter for restricted-GT object-goal episodes.

This module deliberately sits on the *evaluator* side of the V3 boundary.  It
does not import, configure, or modify ``Interactive-Nav-SG-nav``.  A running
ROS graph continues to use its established topics:

* ``/semantic_decision/target`` for a public object-goal context;
* ``/semantic_mapping/gt_observations`` for evaluator-published observations;
* ``/semantic_decision/interaction_command`` for an object-level request; and
* ``/semantic_mapping/interaction_result`` for its minimal completion result.

The perception contract is intentionally small: semantic name, opaque instance
ID, 2D bounding box, segmentation RLE, and 3D axis-aligned box.  The active
wire record is exactly ``{id, name, bbox_2d, mask_rle, box_3d}``, which is the
minimal-GT input understood by the semantic mapping stack.  It deliberately
does not include legacy aliases, derived visibility counters, articulation
metadata, hierarchy, state, or a simulator object name.

Interaction execution is an evaluator-private capability.  A caller registers
``opaque_id -> private_handle`` at reset, consumes an
:class:`EvaluatorInteractionRequest`, invokes its force skill internally, then
calls :meth:`RosObjectGoalEvaluatorAdapter.complete_interaction`.  Published
results expose only command routing, high-level action and success/failure;
they never contain a joint name, joint value, articulation state, or private
handle.

``rospy`` and ``std_msgs`` are imported lazily by :meth:`start`, so this module
can be unit-tested without ROS installed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any

from .restricted_gt_perception import (
    audit_restricted_gt_payload,
    binary_mask_rle_stats,
)


DEFAULT_TARGET_TOPIC = "/semantic_decision/target"
DEFAULT_GT_OBSERVATIONS_TOPIC = "/semantic_mapping/gt_observations"
DEFAULT_INTERACTION_COMMAND_TOPIC = "/semantic_decision/interaction_command"
DEFAULT_INTERACTION_RESULT_TOPIC = "/semantic_mapping/interaction_result"


class RestrictedGTContractError(ValueError):
    """Raised when an evaluator-side payload violates the public contract."""


_PERCEPTION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "episode_id",
        "episode_reset",
        "capture_step",
        "stamp_sec",
        "observations",
    }
)
_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        # Canonical restricted-GT contract.
        "id",
        "name",
        "bbox_2d_xyxy",
        "segmentation_rle",
        "box3d_center",
        "box3d_size",
        # Compatibility aliases for the existing semantic-mapping input
        # normalizer.  They are exact copies of id/name, never private fields.
        "instance_id",
        "semantic_name",
        "source_object_name",
    }
)
_STRICT_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        "id",
        "name",
        "bbox_2d_xyxy",
        "segmentation_rle",
        "box3d_center",
        "box3d_size",
    }
)
_SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS = frozenset(
    {
        "id",
        "name",
        "bbox_2d",
        "mask_rle",
        "box_3d",
    }
)
_SEMANTIC_MINIMAL_BOX_3D_KEYS = frozenset({"center", "size", "frame_id"})
_TARGET_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "episode_id",
        "enabled",
        "target_name",
        "object_labels",
        "instruction",
        # These configure public observation reliability, not task GT.  They
        # are needed because the legacy mapper receives mask-derived counts
        # instead of the old privileged visible-fraction estimate.
        "min_visible_pixels",
        "min_visible_fraction",
        "min_consecutive_observations",
        "completion_requires_visibility",
        "require_current_visibility",
    }
)
_FORBIDDEN_TARGET_CONTEXT_KEYS = frozenset(
    {
        "target_instance_id",
        "target_source_object_name",
        "target_container_instance_id",
        "target_container_source_object_name",
        "target_container_name",
        "target_container_labels",
        "require_interaction",
        "interaction_requirement",
        "oracle_plan",
        "selected_instance",
        "scene_modifications",
    }
)


def adapt_restricted_gt_frame_for_semantic_mapping(
    payload: Mapping[str, Any],
    *,
    stamp_sec: float,
    capture_step: int | None = None,
) -> dict[str, Any]:
    """Convert a V3 restricted-GT frame into the semantic minimal-GT wire form.

    The source frame is independently schema-audited before conversion.  The
    resulting observations retain only the five fields the semantic mapper
    needs: opaque ``id``, semantic ``name``, 2-D box, compact mask RLE and
    world-frame AABB.  In particular, this function must not add legacy
    aliases, decoded pixel coordinates, visibility statistics, object hierarchy
    or any simulator-side articulation metadata.
    """

    audit_restricted_gt_payload(payload)
    observations = [
        {
            "id": str(observation["instance_id"]),
            "name": str(observation["name"]),
            "bbox_2d": list(observation["bbox_2d_xyxy"]),
            "mask_rle": deepcopy(dict(observation["mask_rle"])),
            "box_3d": deepcopy(dict(observation["bbox_3d"])),
        }
        for observation in payload["observations"]
    ]
    adapted = {
        "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["frame_index"] if capture_step is None else capture_step),
        "stamp_sec": float(stamp_sec),
        "observations": observations,
    }
    validate_semantic_minimal_perception_payload(adapted)
    return adapted


def adapt_restricted_gt_frame_for_legacy_mapping(
    payload: Mapping[str, Any],
    *,
    stamp_sec: float,
    capture_step: int | None = None,
    consecutive_observations: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Losslessly bridge the canonical restricted-GT frame to the old mapper.

    The external semantic mapper predates the V3 public protocol and expects
    flattened geometry plus ``source_object_name``.  The latter is *only* an
    opaque-ID alias here.  Every extra field below is either a lossless alias
    of the public 3-D box/name/ID or a deterministic statistic of the public
    segmentation mask; no articulation, hierarchy, state, or source name is
    reintroduced.
    """

    audit_restricted_gt_payload(payload)
    streaks = consecutive_observations if consecutive_observations is not None else {}
    if bool(payload["episode_reset"]):
        streaks.clear()
    visible_now: set[str] = set()
    observations: list[dict[str, Any]] = []
    for observation in payload["observations"]:
        instance_id = str(observation["instance_id"])
        name = str(observation["name"])
        bbox_3d = dict(observation["bbox_3d"])
        mask_rle = dict(observation["mask_rle"])
        _height, _width, visible_pixels = binary_mask_rle_stats(mask_rle)
        visible_now.add(instance_id)
        streaks[instance_id] = int(streaks.get(instance_id, 0)) + 1
        observations.append(
            {
                "instance_id": instance_id,
                "name": name,
                "semantic_name": name,
                # Compatibility only: unchanged ROS code routes interaction
                # commands through this key.  Its value is never a MuJoCo name.
                "source_object_name": instance_id,
                "bbox_2d": list(observation["bbox_2d_xyxy"]),
                "mask_rle": mask_rle,
                "bbox_3d": bbox_3d,
                "position": list(bbox_3d["center"]),
                "aabb_center": list(bbox_3d["center"]),
                "aabb_size": list(bbox_3d["size"]),
                "visible_pixels": visible_pixels,
                "consecutive_observations": int(streaks[instance_id]),
            }
        )
    for instance_id in list(streaks):
        if instance_id not in visible_now:
            streaks.pop(instance_id, None)
    return {
        "schema_version": "interactive_nav_v3_restricted_gt_legacy_adapter_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["frame_index"] if capture_step is None else capture_step),
        "stamp_sec": float(stamp_sec),
        "observations": observations,
    }


def _float_vector(value: Sequence[float] | Any, *, size: int, field_name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RestrictedGTContractError(f"{field_name} must be a numeric {size}-vector") from exc
    if len(result) != size:
        raise RestrictedGTContractError(f"{field_name} must have exactly {size} values")
    return result


def _copy_json_mapping(value: Mapping[str, Any] | None, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RestrictedGTContractError(f"{field_name} must be a mapping")
    result = deepcopy(dict(value))
    try:
        json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise RestrictedGTContractError(f"{field_name} must be JSON serialisable") from exc
    return result


@dataclass(frozen=True)
class RestrictedGTObservation:
    """One visible public instance in the restricted-GT perception contract.

    ``instance_id`` must be a stable, episode-local opaque token such as
    ``"obj_000017"``.  It must not be a MuJoCo body name, asset ID or source
    object name.  ``name`` is a semantic class label (for example ``"door"``
    or ``"refrigerator"``), not an instance identifier.
    """

    instance_id: str
    name: str
    bbox_2d_xyxy: Sequence[float]
    segmentation_rle: Mapping[str, Any] = field(default_factory=dict)
    box3d_center: Sequence[float] = (0.0, 0.0, 0.0)
    box3d_size: Sequence[float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        instance_id = str(self.instance_id).strip()
        name = str(self.name).strip()
        if not instance_id:
            raise RestrictedGTContractError("instance_id must be non-empty")
        if not name:
            raise RestrictedGTContractError("name must be non-empty")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "bbox_2d_xyxy",
            _float_vector(self.bbox_2d_xyxy, size=4, field_name="bbox_2d_xyxy"),
        )
        object.__setattr__(
            self,
            "box3d_center",
            _float_vector(self.box3d_center, size=3, field_name="box3d_center"),
        )
        object.__setattr__(
            self,
            "box3d_size",
            _float_vector(self.box3d_size, size=3, field_name="box3d_size"),
        )
        object.__setattr__(
            self,
            "segmentation_rle",
            _copy_json_mapping(self.segmentation_rle, field_name="segmentation_rle"),
        )

    def to_strict_payload(self) -> dict[str, Any]:
        """Return exactly the six canonical public fields."""

        return {
            "id": self.instance_id,
            "name": self.name,
            "bbox_2d_xyxy": list(self.bbox_2d_xyxy),
            "segmentation_rle": deepcopy(dict(self.segmentation_rle)),
            "box3d_center": list(self.box3d_center),
            "box3d_size": list(self.box3d_size),
        }

    def to_payload(self) -> dict[str, Any]:
        """Return a strict record adapted with lossless legacy aliases."""

        return adapt_strict_observation_for_legacy_mapping(self.to_strict_payload())

    def to_semantic_minimal_payload(self) -> dict[str, Any]:
        """Return the compact record consumed by the semantic mapping stack."""

        return {
            "id": self.instance_id,
            "name": self.name,
            "bbox_2d": list(self.bbox_2d_xyxy),
            "mask_rle": deepcopy(dict(self.segmentation_rle)),
            "box_3d": {
                "center": list(self.box3d_center),
                "size": list(self.box3d_size),
                "frame_id": "world",
            },
        }


def _validate_perception_header(payload: Mapping[str, Any]) -> list[Any]:
    if not isinstance(payload, Mapping):
        raise RestrictedGTContractError("perception payload must be a mapping")
    unexpected_top_level = set(payload) - _PERCEPTION_TOP_LEVEL_KEYS
    if unexpected_top_level:
        raise RestrictedGTContractError(
            "perception payload contains forbidden top-level field(s): "
            + ", ".join(sorted(unexpected_top_level))
        )
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise RestrictedGTContractError("perception payload observations must be a list")
    return observations


def _validate_canonical_observation(observation: Mapping[str, Any], index: int) -> None:
    canonical_id = str(observation.get("id") or "")
    canonical_name = str(observation.get("name") or "")
    if not canonical_id or not canonical_name:
        raise RestrictedGTContractError(f"observation {index} requires non-empty id and name")
    _float_vector(observation.get("bbox_2d_xyxy"), size=4, field_name="bbox_2d_xyxy")
    _float_vector(observation.get("box3d_center"), size=3, field_name="box3d_center")
    _float_vector(observation.get("box3d_size"), size=3, field_name="box3d_size")
    _copy_json_mapping(observation.get("segmentation_rle"), field_name="segmentation_rle")


def validate_strict_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate a frame with only canonical name/id/2D+3D fields.

    This is the transport-neutral contract a detector replacement must satisfy.
    It has no legacy aliases, articulation hints, visibility counters or state.
    """

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _STRICT_PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"strict observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        _validate_canonical_observation(observation, index)


def validate_semantic_minimal_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate the compact minimal-GT payload accepted by semantic mapping.

    This validator intentionally accepts no evaluator compatibility aliases.
    Geometry and masks are copied from the already-audited V3 restricted-GT
    frame, but it also validates direct callers such as a detector-backed
    transport implementation.
    """

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        required = _SEMANTIC_MINIMAL_PERCEPTION_OBSERVATION_KEYS - set(observation)
        if required:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} is missing required field(s): "
                + ", ".join(sorted(required))
            )
        instance_id = str(observation.get("id") or "").strip()
        name = str(observation.get("name") or "").strip()
        if not instance_id or not name:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} requires non-empty id and name"
            )
        _float_vector(observation.get("bbox_2d"), size=4, field_name="bbox_2d")
        try:
            binary_mask_rle_stats(observation["mask_rle"])
        except (TypeError, ValueError) as exc:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} has invalid mask_rle"
            ) from exc
        box_3d = observation.get("box_3d")
        if not isinstance(box_3d, Mapping):
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d must be a mapping"
            )
        unexpected_box_fields = set(box_3d) - _SEMANTIC_MINIMAL_BOX_3D_KEYS
        if unexpected_box_fields:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d contains forbidden field(s): "
                + ", ".join(sorted(unexpected_box_fields))
            )
        required_box_fields = _SEMANTIC_MINIMAL_BOX_3D_KEYS - set(box_3d)
        if required_box_fields:
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d is missing required field(s): "
                + ", ".join(sorted(required_box_fields))
            )
        _float_vector(box_3d.get("center"), size=3, field_name="box_3d.center")
        _float_vector(box_3d.get("size"), size=3, field_name="box_3d.size")
        if not str(box_3d.get("frame_id") or "").strip():
            raise RestrictedGTContractError(
                f"semantic minimal observation {index} box_3d.frame_id must be non-empty"
            )


def adapt_strict_observation_for_legacy_mapping(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly adapt one strict record for the existing mapping input.

    The current ROS semantic mapper reads ``semantic_name`` and
    ``instance_id``.  Its behavior planner also emits ``source_object_name``
    in interaction commands.  All three aliases below are mechanically copied
    from the public ``name``/opaque ``id`` fields, so this function cannot add
    private object identity, ``is_*`` affordances, joints or state.
    """

    validate_strict_perception_payload(
        {
            "schema_version": 1,
            "episode_id": "validation_only",
            "episode_reset": False,
            "capture_step": 0,
            "stamp_sec": 0.0,
            "observations": [dict(observation)],
        }
    )
    strict = deepcopy(dict(observation))
    strict["instance_id"] = strict["id"]
    strict["semantic_name"] = strict["name"]
    strict["source_object_name"] = strict["id"]
    return strict


def adapt_strict_perception_payload_for_legacy_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a legacy-compatible frame without adding any information."""

    validate_strict_perception_payload(payload)
    adapted = deepcopy(dict(payload))
    adapted["observations"] = [
        adapt_strict_observation_for_legacy_mapping(observation)
        for observation in payload["observations"]
    ]
    validate_restricted_perception_payload(adapted)
    return adapted


def adapt_strict_perception_payload_for_semantic_mapping(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert the strict detector-facing form to the active semantic wire form.

    The detector-facing schema intentionally remains independent of the ROS
    mapper.  This conversion is a mechanical field rename plus a fixed public
    world-frame label; it does not derive visibility, aliases or private state.
    """

    validate_strict_perception_payload(payload)
    adapted = {
        "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
        "episode_id": str(payload["episode_id"]),
        "episode_reset": bool(payload["episode_reset"]),
        "capture_step": int(payload["capture_step"]),
        "stamp_sec": float(payload["stamp_sec"]),
        "observations": [
            {
                "id": str(observation["id"]),
                "name": str(observation["name"]),
                "bbox_2d": list(observation["bbox_2d_xyxy"]),
                "mask_rle": deepcopy(dict(observation["segmentation_rle"])),
                "box_3d": {
                    "center": list(observation["box3d_center"]),
                    "size": list(observation["box3d_size"]),
                    "frame_id": "world",
                },
            }
            for observation in payload["observations"]
        ],
    }
    validate_semantic_minimal_perception_payload(adapted)
    return adapted


def validate_restricted_perception_payload(payload: Mapping[str, Any]) -> None:
    """Assert that a legacy-compatible payload remains strictly lossless."""

    observations = _validate_perception_header(payload)
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise RestrictedGTContractError(f"observation {index} must be a mapping")
        unexpected = set(observation) - _PERCEPTION_OBSERVATION_KEYS
        if unexpected:
            raise RestrictedGTContractError(
                f"observation {index} contains forbidden field(s): "
                + ", ".join(sorted(unexpected))
            )
        _validate_canonical_observation(observation, index)
        canonical_id = str(observation.get("id") or "")
        canonical_name = str(observation.get("name") or "")
        if str(observation.get("instance_id") or "") != canonical_id:
            raise RestrictedGTContractError(f"observation {index} instance_id must alias id")
        if str(observation.get("source_object_name") or "") != canonical_id:
            raise RestrictedGTContractError(
                f"observation {index} source_object_name must alias the opaque id"
            )
        if str(observation.get("semantic_name") or "") != canonical_name:
            raise RestrictedGTContractError(f"observation {index} semantic_name must alias name")


@dataclass(frozen=True)
class EvaluatorInteractionRequest:
    """Evaluator-private object-level interaction request.

    ``private_handle`` is intentionally not serialisable and must only be used
    by the evaluator's trusted force-skill executor.  It is never copied into a
    ROS message or trace intended for the evaluated navigation method.
    """

    command_id: str
    episode_id: str
    instance_id: str
    action: str
    private_handle: Any = field(repr=False, compare=False)
    node_id: str = ""
    candidate_id: str = ""
    decision_id: str = ""


InteractionExecutor = Callable[[EvaluatorInteractionRequest], bool | Mapping[str, Any]]


def build_public_target_context(
    *,
    episode_id: str,
    target_name: str,
    instruction: str = "",
    object_labels: Sequence[str] | None = None,
    enabled: bool = True,
    min_visible_pixels: int = 16,
    min_visible_fraction: float = 0.0,
    min_consecutive_observations: int = 1,
    completion_requires_visibility: bool = True,
    require_current_visibility: bool = False,
) -> dict[str, Any]:
    """Build the only target context sent to the ROS navigation method.

    The target class/aliases must come from the public language task (or a
    fixed public language lexicon), never from V3's selected-instance or
    interaction annotations.
    """

    normalized_episode_id = str(episode_id).strip()
    normalized_target_name = str(target_name).strip()
    if not normalized_episode_id:
        raise RestrictedGTContractError("episode_id must be non-empty")
    if not normalized_target_name:
        raise RestrictedGTContractError("target_name must be non-empty")
    labels = [str(item).strip() for item in object_labels or [normalized_target_name]]
    labels = [item for item in labels if item]
    if not labels:
        labels = [normalized_target_name]
    # Preserve supplied order but avoid needless repeated target aliases.
    labels = list(dict.fromkeys(labels))
    context = {
        "schema_version": 1,
        "episode_id": normalized_episode_id,
        "enabled": bool(enabled),
        "target_name": normalized_target_name,
        "object_labels": labels,
        "instruction": str(instruction),
        "min_visible_pixels": max(1, int(min_visible_pixels)),
        "min_visible_fraction": max(0.0, min(1.0, float(min_visible_fraction))),
        "min_consecutive_observations": max(1, int(min_consecutive_observations)),
        "completion_requires_visibility": bool(completion_requires_visibility),
        "require_current_visibility": bool(require_current_visibility),
    }
    validate_public_target_context(context)
    return context


def validate_public_target_context(payload: Mapping[str, Any]) -> None:
    """Reject target contexts containing private V3 task annotations."""

    if not isinstance(payload, Mapping):
        raise RestrictedGTContractError("target context must be a mapping")
    keys = set(payload)
    forbidden = keys & _FORBIDDEN_TARGET_CONTEXT_KEYS
    if forbidden:
        raise RestrictedGTContractError(
            "target context contains private field(s): " + ", ".join(sorted(forbidden))
        )
    unexpected = keys - _TARGET_CONTEXT_KEYS
    if unexpected:
        raise RestrictedGTContractError(
            "target context contains unsupported field(s): " + ", ".join(sorted(unexpected))
        )


class RosObjectGoalEvaluatorAdapter:
    """Bridge a V3 evaluator episode to an unchanged external ROS graph.

    Usage from an evaluator-owned rollout loop::

        adapter.reset(
            episode_id="eval_000042",
            target_context=build_public_target_context(...),
            private_instances={"obj_000017": private_runtime_object},
        )
        adapter.publish_observations(visible_instances, capture_step=0)
        request = adapter.pop_next_interaction_request()
        if request is not None:
            # This call may use joints/force internally; the ROS graph cannot.
            completed = force_skill.open_object(request.private_handle)
            adapter.complete_interaction(request.command_id, success=completed)

    Passing ``interaction_executor`` is useful for a synchronous test harness.
    Production evaluators should normally leave it unset and run force control
    from their own simulation-step loop after consuming pending requests.
    """

    def __init__(
        self,
        *,
        rospy_module: Any | None = None,
        string_message_type: Any | None = None,
        target_topic: str = DEFAULT_TARGET_TOPIC,
        gt_observations_topic: str = DEFAULT_GT_OBSERVATIONS_TOPIC,
        interaction_command_topic: str = DEFAULT_INTERACTION_COMMAND_TOPIC,
        interaction_result_topic: str = DEFAULT_INTERACTION_RESULT_TOPIC,
        interaction_executor: InteractionExecutor | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.target_topic = str(target_topic)
        self.gt_observations_topic = str(gt_observations_topic)
        self.interaction_command_topic = str(interaction_command_topic)
        self.interaction_result_topic = str(interaction_result_topic)
        self._rospy = rospy_module
        self._String = string_message_type
        self._interaction_executor = interaction_executor
        self._clock = clock
        self._lock = threading.RLock()
        self._started = False
        self._target_publisher = None
        self._observations_publisher = None
        self._result_publisher = None
        self._command_subscriber = None
        self._episode_id = ""
        self._private_instances: dict[str, Any] = {}
        self._pending_by_command_id: dict[str, EvaluatorInteractionRequest] = {}
        self._pending_order: deque[str] = deque()
        self._seen_command_ids: set[str] = set()
        self._command_sequence = 0
        self._event_sequence = 0
        self._published_result_events: list[dict[str, Any]] = []
        self._legacy_consecutive_observations: dict[str, int] = {}

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def pending_interaction_count(self) -> int:
        with self._lock:
            return len(self._pending_by_command_id)

    @property
    def published_result_events(self) -> list[dict[str, Any]]:
        """Return public, redacted result events for evaluator diagnostics."""

        with self._lock:
            return deepcopy(self._published_result_events)

    def start(self) -> None:
        """Create ROS publishers/subscriber lazily.

        Supplying ``rospy_module`` and ``string_message_type`` is enough for
        tests; otherwise imports occur only when a live ROS evaluator starts.
        """

        with self._lock:
            if self._started:
                return
            if self._rospy is None or self._String is None:
                try:
                    import rospy  # type: ignore[import-not-found]
                    from std_msgs.msg import String  # type: ignore[import-not-found]
                except ImportError as exc:  # pragma: no cover - exercised only without ROS.
                    raise RuntimeError(
                        "ROS is unavailable; provide rospy_module and string_message_type for tests"
                    ) from exc
                self._rospy = rospy
                self._String = String
            self._target_publisher = self._rospy.Publisher(
                self.target_topic, self._String, queue_size=1, latch=True
            )
            self._observations_publisher = self._rospy.Publisher(
                self.gt_observations_topic, self._String, queue_size=2
            )
            self._result_publisher = self._rospy.Publisher(
                self.interaction_result_topic, self._String, queue_size=4
            )
            self._command_subscriber = self._rospy.Subscriber(
                self.interaction_command_topic,
                self._String,
                self._interaction_command_callback,
                queue_size=8,
            )
            self._started = True

    def close(self) -> None:
        """Detach the command subscriber without mutating the external ROS graph."""

        with self._lock:
            subscriber = self._command_subscriber
            self._command_subscriber = None
            self._started = False
        unregister = getattr(subscriber, "unregister", None)
        if callable(unregister):
            unregister()

    def reset(
        self,
        *,
        episode_id: str,
        target_context: Mapping[str, Any],
        private_instances: Mapping[str, Any],
    ) -> None:
        """Reset public ROS state and replace the evaluator-private ID mapping.

        ``private_instances`` is never serialised.  Its keys must be the same
        opaque IDs that appear in :class:`RestrictedGTObservation` records.
        """

        self.start()
        normalized_episode_id = str(episode_id).strip()
        if not normalized_episode_id:
            raise RestrictedGTContractError("episode_id must be non-empty")
        normalized_target = dict(target_context)
        validate_public_target_context(normalized_target)
        if str(normalized_target.get("episode_id") or "") != normalized_episode_id:
            raise RestrictedGTContractError("target_context episode_id must match reset episode_id")
        normalized_instances: dict[str, Any] = {}
        for public_id, private_handle in private_instances.items():
            opaque_id = str(public_id).strip()
            if not opaque_id:
                raise RestrictedGTContractError("private_instances cannot contain an empty opaque id")
            normalized_instances[opaque_id] = private_handle
        with self._lock:
            self._episode_id = normalized_episode_id
            self._private_instances = normalized_instances
            self._pending_by_command_id.clear()
            self._pending_order.clear()
            self._seen_command_ids.clear()
            self._command_sequence = 0
            self._event_sequence = 0
            self._published_result_events = []
            self._legacy_consecutive_observations.clear()
        # The mapping node resets on this empty episode marker.  Publishing it
        # before target context ensures old graph state cannot be reused.
        self._publish_semantic_minimal_observations_payload(
            {
                "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
                "episode_id": normalized_episode_id,
                "episode_reset": True,
                "capture_step": -1,
                "stamp_sec": float(self._clock()),
                "observations": [],
            }
        )
        self._publish(self._target_publisher, normalized_target)

    def publish_observations(
        self,
        observations: Sequence[RestrictedGTObservation],
        *,
        capture_step: int,
        stamp_sec: float | None = None,
    ) -> dict[str, Any]:
        """Publish a visible-instance frame and return its public payload."""

        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            episode_id = self._episode_id
        semantic_payload = {
            "schema_version": "interactive_nav_v3_semantic_minimal_gt_v1",
            "episode_id": episode_id,
            "episode_reset": False,
            "capture_step": int(capture_step),
            "stamp_sec": float(self._clock() if stamp_sec is None else stamp_sec),
            "observations": [
                observation.to_semantic_minimal_payload() for observation in observations
            ],
        }
        return self.publish_semantic_minimal_perception_payload(semantic_payload)

    def publish_semantic_minimal_perception_payload(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Publish one compact public frame to the semantic mapping stack."""

        validate_semantic_minimal_perception_payload(payload)
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(payload.get("episode_id") or "") != self._episode_id:
                raise RestrictedGTContractError(
                    "semantic minimal perception payload episode_id must match the active episode"
                )
        self._publish_semantic_minimal_observations_payload(payload)
        return deepcopy(dict(payload))

    def publish_strict_perception_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Receive a canonical strict frame and publish its semantic adapter.

        This is the entry point for future detector-backed perception.  The
        conversion adds only lossless field aliases required by the semantic
        minimal-GT input;
        :func:`validate_strict_perception_payload` rejects ``is_*``, joint,
        state, pose, relation and visibility fields before publication.
        """

        validate_strict_perception_payload(payload)
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(payload.get("episode_id") or "") != self._episode_id:
                raise RestrictedGTContractError(
                    "strict perception payload episode_id must match the active episode"
                )
        adapted = adapt_strict_perception_payload_for_semantic_mapping(payload)
        self._publish_semantic_minimal_observations_payload(adapted)
        return deepcopy(adapted)

    def publish_restricted_gt_frame(
        self,
        payload: Mapping[str, Any],
        *,
        capture_step: int | None = None,
        stamp_sec: float | None = None,
    ) -> dict[str, Any]:
        """Publish the canonical V3 restricted-GT payload to semantic mapping.

        The canonical payload is first audited by
        :mod:`restricted_gt_perception`; the wire message has exactly the
        semantic minimal-GT fields ``id/name/bbox_2d/mask_rle/box_3d`` per
        observation.  It never restores object source names, legacy aliases,
        decoded mask coordinates, articulation metadata, state, relations, or
        camera privilege.
        """

        # Convert and audit before acquiring the ROS-state lock, avoiding a
        # redundant outer payload audit on every frame.
        adapted = adapt_restricted_gt_frame_for_semantic_mapping(
            payload,
            capture_step=capture_step,
            stamp_sec=float(self._clock() if stamp_sec is None else stamp_sec),
        )
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before publishing observations")
            if str(adapted["episode_id"]) != self._episode_id:
                raise RestrictedGTContractError(
                    "restricted-GT frame episode_id must match the active episode"
                )
        self._publish_semantic_minimal_observations_payload(adapted)
        return deepcopy(adapted)

    def pop_next_interaction_request(self) -> EvaluatorInteractionRequest | None:
        """Return one pending private request for evaluator-side force control."""

        with self._lock:
            while self._pending_order:
                command_id = self._pending_order.popleft()
                request = self._pending_by_command_id.get(command_id)
                if request is not None:
                    return request
        return None

    def complete_interaction(
        self,
        command_id: str,
        *,
        success: bool,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Publish a minimal high-level completion result for one request.

        Force-policy outputs must be reduced to the boolean ``success`` before
        this method is called.  In particular, do not pass joint diagnostics or
        a post-action state through the ROS result topic.
        """

        with self._lock:
            request = self._pending_by_command_id.pop(str(command_id), None)
        if request is None:
            raise KeyError(f"No pending interaction command {command_id!r}")
        return self._emit_interaction_result(
            request,
            success=bool(success),
            status=status or ("SUCCEEDED" if success else "FAILED"),
        )

    def reject_interaction(
        self,
        command_id: str,
        *,
        reason: str = "rejected",
    ) -> dict[str, Any]:
        """Finish a pending request without exposing evaluator diagnostics."""

        with self._lock:
            request = self._pending_by_command_id.pop(str(command_id), None)
        if request is None:
            raise KeyError(f"No pending interaction command {command_id!r}")
        return self._emit_interaction_result(
            request,
            success=False,
            status="REJECTED",
            reason=str(reason),
        )

    def receive_interaction_command(
        self, payload: Mapping[str, Any]
    ) -> EvaluatorInteractionRequest | None:
        """Accept a ROS command without trusting its object/joint selectors.

        Only an opaque ID registered for the current episode can resolve to a
        private object.  Incoming ``joint_names``, force settings, or guessed
        source names are deliberately ignored and never reach the executor.
        """

        if not isinstance(payload, Mapping):
            raise RestrictedGTContractError("interaction command must be a mapping")
        with self._lock:
            if not self._episode_id:
                raise RuntimeError("reset() must be called before accepting interaction commands")
            self._command_sequence += 1
            command_id = str(payload.get("command_id") or "").strip()
            if not command_id:
                command_id = f"object_skill_{self._command_sequence:06d}"
            if command_id in self._seen_command_ids:
                return None
            self._seen_command_ids.add(command_id)
            instance_id = self._command_instance_id(payload)
            action = str(payload.get("action") or "open").strip().casefold()
            node_id = str(payload.get("node_id") or "")
            candidate_id = str(payload.get("candidate_id") or "")
            decision_id = str(payload.get("decision_id") or "")
            private_handle = self._private_instances.get(instance_id)
            episode_id = self._episode_id
        if action != "open":
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                instance_id=instance_id,
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="unsupported_action",
            )
        if private_handle is None:
            return self._reject_unresolved_command(
                command_id=command_id,
                episode_id=episode_id,
                instance_id=instance_id,
                action=action,
                node_id=node_id,
                candidate_id=candidate_id,
                decision_id=decision_id,
                reason="unknown_instance_id",
            )
        request = EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id=episode_id,
            instance_id=instance_id,
            action=action,
            private_handle=private_handle,
            node_id=node_id,
            candidate_id=candidate_id,
            decision_id=decision_id,
        )
        with self._lock:
            self._pending_by_command_id[command_id] = request
            self._pending_order.append(command_id)
        if self._interaction_executor is not None:
            try:
                outcome = self._interaction_executor(request)
                success = bool(outcome.get("success")) if isinstance(outcome, Mapping) else bool(outcome)
            except Exception:
                # The external method gets only a generic failure signal.  Full
                # exception details remain evaluator diagnostics.
                success = False
            self.complete_interaction(command_id, success=success)
            return None
        return request

    def _interaction_command_callback(self, message: Any) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise RestrictedGTContractError("interaction command must decode to an object")
            self.receive_interaction_command(payload)
        except (TypeError, ValueError, json.JSONDecodeError, RuntimeError):
            # A malformed command has no safe, routable command ID to respond
            # to.  Ignore it rather than leaking evaluator state on a log topic.
            return

    @staticmethod
    def _command_instance_id(payload: Mapping[str, Any]) -> str:
        # The semantic minimal-GT graph uses ``object_id`` for the same opaque
        # token that the evaluator registered at reset.  Legacy aliases remain
        # accepted for older standalone methods, but all values are resolved
        # only against the evaluator-private opaque-ID registry below.
        for key in ("object_id", "instance_id", "source_object_name", "target_instance_id"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return ""

    def _reject_unresolved_command(
        self,
        *,
        command_id: str,
        episode_id: str,
        instance_id: str,
        action: str,
        node_id: str,
        candidate_id: str,
        decision_id: str,
        reason: str,
    ) -> None:
        request = EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id=episode_id,
            instance_id=instance_id,
            action=action,
            private_handle=None,
            node_id=node_id,
            candidate_id=candidate_id,
            decision_id=decision_id,
        )
        self._emit_interaction_result(
            request,
            success=False,
            status="REJECTED",
            reason=reason,
        )
        return None

    def _emit_interaction_result(
        self,
        request: EvaluatorInteractionRequest,
        *,
        success: bool,
        status: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._event_sequence += 1
            event_id = f"object_skill_{self._event_sequence:06d}"
        # The output has deliberately no state, joint, or force fields.  A
        # method can observe the consequence only through the next public
        # perception frame, while the evaluator separately retains full V3
        # completion checks for final scoring.
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_id": event_id,
            "episode_id": request.episode_id,
            "command_id": request.command_id,
            "candidate_id": request.candidate_id,
            "decision_id": request.decision_id,
            "node_id": request.node_id,
            # ``object_id`` is the semantic graph's public opaque identifier;
            # it is copied from the command and never resolved back to a
            # simulator name on this ROS topic.
            "object_id": request.instance_id,
            "instance_id": request.instance_id,
            "action": request.action,
            "success": bool(success),
            "status": str(status),
            "source": "evaluator_object_skill",
            "stamp_sec": float(self._clock()),
        }
        if reason:
            payload["reason"] = str(reason)
        self._publish(self._result_publisher, payload)
        with self._lock:
            self._published_result_events.append(deepcopy(payload))
        return payload

    def _publish_observations_payload(self, payload: Mapping[str, Any]) -> None:
        validate_restricted_perception_payload(payload)
        self._publish(self._observations_publisher, dict(payload))

    def _publish_semantic_minimal_observations_payload(self, payload: Mapping[str, Any]) -> None:
        validate_semantic_minimal_perception_payload(payload)
        self._publish(self._observations_publisher, dict(payload))

    def _publish(self, publisher: Any, payload: Mapping[str, Any]) -> None:
        if publisher is None or self._String is None:
            raise RuntimeError("start() must be called before publishing ROS messages")
        encoded = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        publisher.publish(self._String(data=encoded))
