from __future__ import annotations

import time
import math
from collections import defaultdict

import numpy as np

from .geometry_utils import (
    grid_index,
    grid_origin_yaw,
    grid_to_world,
    normalize_label,
    world_to_grid,
)
from .graph_rules import (
    PORTAL_LABELS,
    default_interaction_payload,
    distance_xy,
    infer_node_type,
    is_public_door_id,
    normalize_observation,
    room_node_label,
    sanitize_token,
)
from .graph_schema import NavigationHint, SceneGraphBundle, SceneGraphEdge, SceneGraphNode
from .room_inference_backends import WeightedRoomAttributeInferencer
from .semantic_evidence import (
    _portal_geometry_plausible,
    _portal_attrs_geometry_plausible,
    _accepted_semantic_evidence,
    _has_m1_portal_confirmation,
    persistence_action,
)


_SEMANTIC_POSTCONDITION_BY_ACTION = {
    "open": "open",
    "close": "closed",
}


# Room-segmentation IDs are allocated from small positive integers.  Keep
# portal-child rooms in a separate, stable range: they represent a topological
# hypothesis only and must never be written into the occupancy-derived room
# grid.
_PORTAL_CHILD_ROOM_ID_BASE = 1_000_000
_PARENT_CACHE_MISS = object()
_PORTAL_MATCH_LABELS = frozenset(
    {"portal", "door", "doorway", "doorframe", "door_leaf", "gate", "entrance"}
)

# These are semantic landmarks rather than short-lived detector tracks.  Once
# the temporal detector gate and the visual Module-1 gate have both passed,
# their identity must survive occlusion, viewpoint changes, and detector
# dropouts.  Visibility and geometry continue to be refreshed separately.
_PERSISTENT_SEMANTIC_TYPES = frozenset({"room", "portal", "container"})

# Label predicates are evaluated repeatedly while refreshing every graph node
# (especially the refrigerator/content relation fallback).  Labels come from
# a bounded open-vocabulary detector stream, so a small explicit cache avoids
# re-normalizing the same strings without retaining an unbounded history.
_LABEL_MARKER_CACHE_MAX = 4096
_LABEL_MARKER_CACHE = {}
_REFRIGERATOR_LABEL_CACHE = {}


def _cached_label_marker_match(label, marker):
    """Match already-normalized labels with a bounded pair cache."""

    key = (label, marker)
    cached = _LABEL_MARKER_CACHE.get(key)
    if cached is not None:
        return cached
    result = bool(
        label
        and marker
        and (
            label == marker
            or label.startswith(f"{marker}_")
            or label.endswith(f"_{marker}")
        )
    )
    if len(_LABEL_MARKER_CACHE) >= _LABEL_MARKER_CACHE_MAX:
        _LABEL_MARKER_CACHE.clear()
    _LABEL_MARKER_CACHE[key] = result
    return result


_SEMANTIC_CONFIRMATION_FIELDS = (
    "attribute_status", "attribute_confidence", "mllm_interaction_class",
    "m1_observed_object_name", "m1_refrigerator_pending_confirmation", "attribute_updated_at",
)


def _has_persistent_semantic_evidence(node):
    """Return whether a node has passed the two-frame + M1 admission gate."""

    if node is None or node.type not in _PERSISTENT_SEMANTIC_TYPES:
        return False
    if not bool(node.attributes.get("persistent_semantic_node", False)):
        return False
    return node.type != "portal" or _has_m1_portal_confirmation(node.attributes)


def _resolved_interaction_state(result):
    """Resolve a public semantic postcondition without reading joint state.

    Evaluator-owned interaction skills intentionally expose only the requested
    action and whether the sealed skill succeeded.  In that contract a
    successful ``open``/``close`` result establishes the corresponding
    semantic postcondition even when no simulator articulation is returned.
    """

    # A failed action may still carry the controller's last/desired
    # ``post_state`` (for example ``open`` after a rejected M3 request).  It is
    # not evidence that the object reached that state.  Portal results are
    # additionally checked by ``_portal_result_state_gate``; keep this guard
    # here as well so containers cannot be promoted by a failed/stale result.
    if result.get("success") is False:
        return None, False
    explicit_state = result.get("state") or result.get("post_state")
    if explicit_state is not None:
        return str(explicit_state), False
    if result.get("success") is not True:
        return None, False
    action = str(result.get("action") or "").strip().casefold()
    inferred_state = _SEMANTIC_POSTCONDITION_BY_ACTION.get(action)
    if inferred_state is None and not action:
        expected_state = str(result.get("expected_state") or "").strip().casefold()
        if expected_state in {"open", "closed"}:
            inferred_state = expected_state
    return inferred_state, inferred_state is not None


def _interaction_result_capability(result):
    """Classify executor feedback without exposing simulator joint metadata."""

    capability = str(result.get("interaction_capability") or "").strip().casefold()
    state = str(result.get("state") or result.get("post_state") or "").strip().casefold()
    if (
        str(result.get("observation_outcome") or "").strip().casefold()
        == "finish_without_action"
        and capability == "unavailable"
    ):
        return "unavailable"
    if (
        capability == "static"
        or state in {"static", "static_open", "static_closed"}
    ):
        return "static"
    if capability == "unavailable" or state == "unavailable":
        # A missing articulation identifies an unavailable executor target,
        # not a fixed open passage.  Preserve that distinction so later visual
        # observations cannot turn a failed action into ``static_open``.
        return "unavailable"
    if (
        capability in {"blocked", "unsupported", "locked"}
        or state == "blocked"
        or result.get("interactable") is False
    ):
        # A non-static negative result is an executor/MLLM statement that this
        # interaction cannot currently be used, rather than a claim that the
        # observed portal is merely an open doorway.
        return "blocked"
    return "unknown"


def _public_portal_morphology(patch):
    """Return a compact visual-only portal morphology, if supplied.

    Attribute updates normally pass through the MLLM schema validator, but the
    graph store also accepts replay/test payloads directly.  Normalize again at
    this boundary so no arbitrary model text becomes part of the public graph.
    """

    raw = patch.get("portal_morphology")
    if not isinstance(raw, dict):
        return None
    leaf = str(raw.get("door_leaf") or raw.get("leaf") or "unknown").strip().casefold()
    if leaf in {"absent", "none", "missing", "no_leaf", "no_door"}:
        leaf = "absent"
    elif leaf in {"present", "leaf", "door", "door_leaf", "visible"}:
        leaf = "present"
    else:
        leaf = "unknown"
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"door_leaf": leaf, "confidence": confidence}


def _public_portal_aperture_evidence(patch):
    """Return the bounded, visual-only opening evidence from a MLLM patch."""

    raw = patch.get("portal_aperture_evidence")
    if not isinstance(raw, dict):
        return None
    aperture = str(
        raw.get("open_aperture") or raw.get("aperture") or raw.get("opening") or "unknown"
    ).strip().casefold()
    if raw.get("opening_visible") is True or raw.get("aperture_open") is True:
        aperture = "visible"
    elif raw.get("opening_visible") is False or raw.get("aperture_open") is False:
        aperture = "not_visible"
    if aperture in {"visible", "open", "opening_visible", "clear_gap", "gap"}:
        aperture = "visible"
    elif aperture in {"not_visible", "closed", "occluded", "no_gap", "not_open"}:
        aperture = "not_visible"
    else:
        aperture = "unknown"
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {"open_aperture": aperture, "confidence": confidence}


def _m1_observed_object_name(patch):
    """Return M1's concrete visual name, ignoring empty/generic answers."""

    raw = str(
        patch.get("observed_object_name")
        or patch.get("m1_object_name")
        or patch.get("object_name")
        or ""
    ).strip()
    normalized = normalize_label(raw)
    if not normalized or normalized in {
        "object",
        "thing",
        "item",
        "unknown",
        "none",
        "container",
        "portal",
    }:
        return ""
    return normalized[:64]


def _remember_m1_refrigerator_reference(node, observed_name, confidence):
    """Persist a body reference when M1 promotes a generic crop to a fridge.

    The detector normally supplies the appliance class before M1 runs, but a
    physical crop can be emitted as a generic ``object``.  In that case the
    first M1 answer is the only refrigerator identity available.  Preserve a
    stable label/box at that point so a later detector frame cannot erase the
    identity before the sealed open result is rebuilt into relations.  Once an
    interaction has succeeded, the reference is intentionally frozen: an open
    door often changes the live box to include the leaf and visible contents.
    """

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return False
    if confidence < 0.5 or not _is_refrigerator_label(observed_name):
        return False
    attributes = node.attributes or {}
    normalized_name = normalize_label(observed_name)
    changed = False
    if not _is_refrigerator_label(attributes.get("interaction_reference_label")):
        attributes["interaction_reference_label"] = normalized_name
        changed = True

    center = getattr(node, "aabb_center", None)
    size = getattr(node, "aabb_size", None)
    if _valid_float_vector(center, 3) and _valid_float_vector(size, 3):
        try:
            current_size = [abs(float(size[index])) for index in range(3)]
        except (IndexError, TypeError, ValueError):
            current_size = []
        valid_size = len(current_size) == 3 and all(
            math.isfinite(value) and value > 1e-6 for value in current_size
        )
        existing_size = attributes.get("interaction_reference_aabb_size")
        reference_valid = _valid_float_vector(existing_size, 3)
        # Do not replace a configured/stable reference after a successful
        # action.  Before an action, replace only an absent/invalid or clearly
        # tiny reference with a plausible body-sized crop.
        try:
            existing_body_sized = bool(
                reference_valid
                and max(abs(float(existing_size[0])), abs(float(existing_size[1])))
                >= 0.35
                and abs(float(existing_size[2])) >= 0.65
                and abs(float(existing_size[0]))
                * abs(float(existing_size[1]))
                * abs(float(existing_size[2]))
                >= 0.18
            )
        except (IndexError, TypeError, ValueError):
            existing_body_sized = False
        interaction_confirmed = _has_confirmed_refrigerator_interaction(node)
        configured_reference = str(
            attributes.get("interaction_geometry_source") or ""
        ).strip().casefold() == "configured_override"
        if valid_size and (
            not reference_valid
            or (
                not interaction_confirmed
                and not configured_reference
                and not existing_body_sized
            )
        ):
            attributes["interaction_reference_aabb_center"] = [
                float(center[index]) for index in range(3)
            ]
            attributes["interaction_reference_aabb_size"] = current_size
            changed = True
    yaw = attributes.get("yaw")
    if not _valid_float_scalar(attributes.get("interaction_reference_yaw")):
        try:
            yaw = float(yaw)
        except (TypeError, ValueError):
            yaw = None
        if yaw is not None and math.isfinite(yaw):
            attributes["interaction_reference_yaw"] = yaw
            changed = True
    attributes["m1_refrigerator_evidence"] = True
    attributes["m1_refrigerator_evidence_confidence"] = max(
        float(attributes.get("m1_refrigerator_evidence_confidence", 0.0) or 0.0),
        confidence,
    )
    return changed


def _container_m1_hysteresis(
    node,
    interaction_class,
    requested_state,
    requested_interactable,
    *,
    is_visual_mllm_patch,
):
    """Keep a source-established container stable across one M1 response.

    Module 1 classifies an image crop and is allowed to be uncertain about the
    object ontology.  A one-frame ``portal``/``none`` answer must therefore not
    rewrite a graph node that the observation lane already established as a
    container.  The same applies to ``static_open``: that state is a topology
    terminal for portals, but is not a valid one-frame terminal for a fridge or
    drawer.  We deliberately do not infer simulator capability or an approach
    direction here; an uncertain container remains an ordinary interaction
    candidate and asks the caller for a fresh visual observation.

    Returns ``(type_locked, state_rejected, reason)``.  The graph records the
    rejection for diagnostics while retaining the previous interaction payload.
    A future policy may add a multi-frame, public-evidence promotion lane, but
    this conservative lane never promotes a container from one M1 frame.
    """

    if not is_visual_mllm_patch:
        return False, False, "not_visual_mllm_patch"
    attributes = node.attributes or {}
    topology_type = str(
        attributes.get("topology_type")
        or attributes.get("observation_node_type")
        or ""
    ).strip().casefold()
    established_container = node.type == "container" or topology_type == "container"
    if not established_container:
        return False, False, "not_established_container"

    requested_class = str(interaction_class or "unknown").strip().casefold()
    requested_state = str(requested_state or "unknown").strip().casefold()
    type_locked = requested_class != "container"
    # A negative image-only capability claim is not a terminal result for an
    # otherwise openable source container.  Preserve its existing candidate
    # and ask for another view; sealed executor feedback remains the only lane
    # allowed to mark an openable container unavailable.
    noninteractable_rejected = bool(
        requested_class == "container"
        and not bool(requested_interactable)
        and bool(node.interaction.get("is_interactable"))
    )
    state_rejected = (
        type_locked
        or requested_state == "static_open"
        or noninteractable_rejected
    )
    if not state_rejected:
        return False, False, "container_evidence_accepted"
    reasons = []
    if type_locked:
        reasons.append(f"class:{requested_class or 'unknown'}")
    if requested_state == "static_open":
        reasons.append("state:static_open")
    if noninteractable_rejected:
        reasons.append("interactable:false")
    return (
        type_locked,
        state_rejected,
        ";".join(reasons) or "uncertain_container_evidence",
    )


def _portal_m1_type_hysteresis(node, interaction_class, *, is_visual_mllm_patch):
    """Keep a topology-established portal stable across one visual response."""

    if not is_visual_mllm_patch:
        return False, "not_visual_mllm_patch"
    attributes = node.attributes or {}
    topology_type = str(
        attributes.get("topology_type")
        or attributes.get("observation_node_type")
        or ""
    ).strip().casefold()
    established_portal = node.type == "portal" or topology_type == "portal"
    if not established_portal:
        return False, "not_established_portal"
    requested_class = str(interaction_class or "unknown").strip().casefold()
    if requested_class == "portal":
        return False, "portal_evidence_accepted"
    return True, f"class:{requested_class or 'unknown'}"


def _portal_has_observed_open_connectivity(node):
    """Whether mapping, not a synthetic post-open hypothesis, saw both sides."""

    attributes = node.attributes or {}
    observed_room_ids = {
        str(room_id)
        for room_id in (attributes.get("observed_connected_room_ids") or [])
        if room_id is not None and str(room_id)
    }
    return bool(
        node.is_currently_visible
        and str(attributes.get("connectivity_status") or "").casefold()
        == "connected"
        and len(observed_room_ids) >= 2
        and not bool(attributes.get("potential_room_ids"))
    )


def _portal_has_visual_and_map_open_evidence(node, aperture_evidence):
    """Whether independent M1 aperture and occupancy evidence agree on open."""

    aperture_visible = bool(
        isinstance(aperture_evidence, dict)
        and aperture_evidence.get("open_aperture") == "visible"
        and float(aperture_evidence.get("confidence", 0.0) or 0.0) >= 0.70
    )
    return bool(aperture_visible and _portal_has_observed_open_connectivity(node))


def _portal_visual_state_gate(
    node,
    state,
    morphology,
    aperture_evidence,
    *,
    visual_evidence_truncated=False,
):
    """Gate MLLM portal state claims on direct visual aperture evidence.

    A visual class can identify a door but cannot by itself establish that its
    leaf moved.  This prevents a single hallucinated ``open`` from suppressing
    the required first interaction.  ``static_open`` has the stricter
    morphology + independently observed map-connectivity gate.
    """

    requested = str(state or "unknown").strip().casefold()
    if (
        node.type == "portal"
        and requested in {"open", "ajar", "static_open", "closed"}
        and bool(visual_evidence_truncated)
    ):
        # Border-clipped evidence is incomplete in both directions: it cannot
        # prove an aperture and it cannot prove that the whole leaf is closed.
        return False, "truncated_visual_evidence"
    if node.type != "portal" or requested not in {"open", "ajar", "static_open"}:
        return True, "not_applicable"
    aperture_visible = bool(
        isinstance(aperture_evidence, dict)
        and aperture_evidence.get("open_aperture") == "visible"
        and float(aperture_evidence.get("confidence", 0.0) or 0.0) >= 0.70
    )
    if requested in {"open", "ajar"}:
        # M1's ``open`` result is a direct visual state observation.  Once the
        # crop contains a confident visible aperture, the door interaction is
        # already satisfied and must leave the interaction-candidate pool.
        # Room/OCC connectivity is still used by the traversal planner, but it
        # must not keep an already-open door as a repeated ``open`` subgoal.
        map_connected = _portal_has_observed_open_connectivity(node)
        return (
            aperture_visible,
            "visual_open_aperture_and_map_confirmed"
            if aperture_visible and map_connected
            else "missing_visual_open_aperture_evidence"
            if not aperture_visible
            else "visual_open_aperture_confirmed",
        )
    fixed_opening = bool(
        isinstance(morphology, dict)
        and morphology.get("door_leaf") == "absent"
        and float(morphology.get("confidence", 0.0) or 0.0) >= 0.70
        and aperture_visible
        and _portal_has_observed_open_connectivity(node)
    )
    return (
        fixed_opening,
        "fixed_opening_visual_and_map_confirmed"
        if fixed_opening
        else "missing_fixed_opening_evidence",
    )


def _is_confirmed_portal_open_result(result, resolved_state):
    """Gate synthetic post-open topology on a real confirmed open result.

    A successful opaque ``open`` result is accepted as a public postcondition,
    but explicit static/blocked capability or state always vetoes it.  In
    particular, ``static_open`` means an already-open passage and must not
    manufacture a far-side room merely because an interaction command was
    issued.
    """

    if result.get("success") is not True:
        return False
    if str(result.get("action") or "").strip().casefold() != "open":
        return False
    state = str(resolved_state or "").strip().casefold()
    capability = str(
        result.get("interaction_capability") or result.get("capability") or ""
    ).strip().casefold()
    source = str(result.get("source") or "").strip().casefold()
    if state in {"static", "static_open", "static_closed"}:
        return False
    if capability in {
        "static",
        "blocked",
        "unsupported",
        "unavailable",
        "locked",
    }:
        return False
    if source == "executor_static_portal":
        return False
    return state in {"open", "opened"}


def _portal_result_state_gate(node, result, resolved_state):
    """Allow result-lane portal opening only after a real successful action."""

    requested = str(resolved_state or "").strip().casefold()
    if node.type != "portal" or requested not in {
        "open",
        "opened",
        "ajar",
        "static_open",
    }:
        return True, "not_applicable"
    if result.get("success") is False:
        return False, "unsuccessful_result"
    if bool(
        result.get("visual_evidence_truncated")
        or result.get("visual_evidence_truncated_edges")
    ):
        # A result may be correlated with a real observation, but an image
        # clipped at the frame boundary cannot establish either an open gap or
        # a closed leaf.  Preserve the prior graph state until a complete view
        # or an authoritative simulator result arrives.
        return False, "truncated_visual_evidence"
    action = str(result.get("action") or "").strip().casefold()
    capability = _interaction_result_capability(result)
    source = str(result.get("source") or "").strip().casefold()
    observation_only = (
        result.get("success") is True
        and str(result.get("observation_outcome") or "").strip().casefold()
        == "finish_without_action"
    )
    trusted_verified_state = source in {
        "oracle_interaction",
        "direct_joint_readback",
        "executor_state_verification",
        "successful_action_postcondition",
    }
    static_terminal = (
        requested == "static_open"
        and capability == "static"
        and result.get("success") is True
    )
    accepted = bool(
        static_terminal
        or (
            observation_only
            and requested == "static_open"
            and capability in {"unknown", "unavailable"}
            and _portal_has_visual_and_map_open_evidence(
                node,
                result.get("portal_aperture_evidence"),
            )
        )
        or trusted_verified_state
        or (
            result.get("success") is True
            and action == "open"
            and requested in {"open", "opened", "ajar"}
            and capability not in {"static", "blocked", "unavailable"}
            and source != "executor_static_portal"
        )
    )
    return (
        accepted,
        "trusted_verified_state"
        if trusted_verified_state
        else "successful_force_open"
        if accepted and not static_terminal
        else "static_terminal"
        if static_terminal
        else "missing_successful_force_open",
    )


class InteractionGraphStore:
    def __init__(
        self,
        scene_id="scene",
        match_distance=0.5,
        room_id_to_name=None,
        room_box_height=0.2,
        portal_room_max_radius_m=1.0,
        object_room_search_margin_m=0.75,
        object_room_priors=None,
        room_attribute_min_confidence=0.2,
        room_mllm_min_confidence=0.55,
        interaction_geometry_overrides=None,
        portal_child_room_enabled=True,
        portal_child_room_offset_m=0.9,
        portal_child_room_depth_m=1.2,
        portal_child_room_min_width_m=1.2,
    ):
        self.scene_id = str(scene_id or "scene")
        self.match_distance = float(match_distance)
        self.room_id_to_name = dict(room_id_to_name or {})
        self.room_box_height = float(room_box_height)
        self.portal_room_max_radius_m = max(0.1, float(portal_room_max_radius_m))
        self.object_room_search_margin_m = max(
            0.1, float(object_room_search_margin_m)
        )
        self.room_attribute_inferencer = WeightedRoomAttributeInferencer(
            object_room_priors or {},
            min_confidence=room_attribute_min_confidence,
        )
        # The rule inferencer remains an immediate fallback.  An asynchronous
        # MLLM room result is allowed to replace it only when it is both fresh
        # and confident enough; otherwise every relation rebuild would erase a
        # useful result or let an uncertain answer steer exploration.
        self.room_mllm_min_confidence = max(
            0.0, min(1.0, float(room_mllm_min_confidence))
        )
        self.interaction_geometry_overrides = {
            str(key): dict(value or {})
            for key, value in (interaction_geometry_overrides or {}).items()
        }
        self.portal_child_room_enabled = bool(portal_child_room_enabled)
        self.portal_child_room_offset_m = max(
            0.2, float(portal_child_room_offset_m)
        )
        self.portal_child_room_depth_m = max(
            0.4, float(portal_child_room_depth_m)
        )
        self.portal_child_room_min_width_m = max(
            0.4, float(portal_child_room_min_width_m)
        )
        self.room_geometries = {}
        self.room_geometry_candidates = {}
        self._room_split_shrink_allowed = set()
        self.room_geometry_stability_frames = 5
        self.room_redirects = {}
        self.nodes = {}
        # Detector observations normally carry a stable instance/track ID.
        # Keep a per-update identity index so matching does not scan every
        # historical graph node for each 10 Hz observation.  The index is
        # rebuilt at the start of an observation batch and updated as nodes
        # are observed/created; direct node mutations are outside the store's
        # observation API and are reconciled on the next batch boundary.
        self._identity_index = None
        # Spatial buckets are the secondary index for observations that do
        # not carry a stable detector identity.  The historical matcher
        # scanned every object of the same label/type, which made a large
        # first batch quadratic even when detections were far apart.
        self._spatial_index = None
        self._spatial_node_bucket = {}
        self._spatial_node_order = {}
        self._spatial_cell_size = max(self.match_distance, 0.1)
        self._parent_relation_cache = {}
        self._parent_relation_cache_context = None
        self.edges = {}
        self.next_node_index = 1
        self.edge_counter = 1
        self.room_grid = None
        # Statistics for the most recently accepted room grid.  Room
        # segmentation commonly republishes the same label/confidence arrays
        # for each newer OCC source.  Keeping the derived per-room scalars
        # behind this private seam lets ``update_room_grid`` advance graph
        # revision/relations without rescanning every cell on a cache hit.
        self._room_grid_stats_cache = None
        self.last_room_grid_cache_hit = False
        # Portal-to-room probing is pure with respect to one accepted room
        # grid and one portal geometry.  Detection observations arrive more
        # frequently than room segmentation, so cache the result between
        # those observations and invalidate it only when the grid topology
        # (or a room merge) advances.
        self._room_grid_epoch = 0
        self._portal_room_ids_cache: dict[tuple, tuple[int, ...]] = {}
        self.source_mode = "detector_online"
        self.episode_id = ""
        self.graph_revision = 0
        self.capture_step = None
        self.interaction_event_counter = 1
        self.next_portal_child_room_id = _PORTAL_CHILD_ROOM_ID_BASE
        self.force_provisional_portal_rooms = True
        self._ensure_scene_node()

    def reset(self, episode_id="", source_mode=None):
        self.episode_id = str(episode_id or "")
        if source_mode:
            self.source_mode = str(source_mode)
        self.room_geometries = {}
        self.room_geometry_candidates = {}
        self._room_split_shrink_allowed = set()
        self.room_geometry_stability_frames = 5
        self.room_redirects = {}
        self.nodes = {}
        self._identity_index = None
        self._spatial_index = None
        self._spatial_node_bucket = {}
        self._spatial_node_order = {}
        self._parent_relation_cache = {}
        self._parent_relation_cache_context = None
        self.edges = {}
        self.next_node_index = 1
        self.edge_counter = 1
        self.room_grid = None
        self._room_grid_stats_cache = None
        self.last_room_grid_cache_hit = False
        self._room_grid_epoch = 0
        self._portal_room_ids_cache = {}
        self.graph_revision = 0
        self.capture_step = None
        self.interaction_event_counter = 1
        self.next_portal_child_room_id = _PORTAL_CHILD_ROOM_ID_BASE
        self._ensure_scene_node()

    def has_confirmed_open_refrigerator(self):
        """Return whether the graph contains a trusted open refrigerator.

        This small read-only seam is used by the mapper before exporting
        short-lived detector tracks.  It deliberately re-applies the narrow
        refrigerator-type restoration rule first: a delayed M1 ``object``
        label must not hide an appliance whose successful open result is
        already sealed in ``operation_history``.  No visual M1-only state is
        accepted as an open transition.
        """

        for node in self.nodes.values():
            if node.type not in {"container", "object"}:
                continue
            _restore_confirmed_refrigerator_container_type(node)
            if node.type == "container" and _is_open_refrigerator(node):
                return True
        return False

    def update_room_grid(
        self,
        grid_info,
        scene_data,
        confidence_data=None,
        room_id_to_name=None,
        room_merges=None,
        geometry_stability_frames=5,
    ):
        self.room_geometry_stability_frames = max(
            1, int(geometry_stability_frames)
        )
        if room_id_to_name:
            self.room_id_to_name.update({int(k): str(v) for k, v in room_id_to_name.items()})

        # ``scene_data`` and ``confidence_data`` are normally fresh Python
        # lists, even when their values are unchanged from the previous OCC
        # frame.  Compare them before copying; list equality is implemented in
        # C and is materially cheaper than the NumPy/statistics pass below.
        # A non-empty merge changes redirects and therefore must take the full
        # path once so cached room statistics cannot use stale resolved IDs.
        previous_grid = self.room_grid
        cache_hit = bool(
            not room_merges
            and self._room_grid_stats_cache is not None
            and previous_grid is not None
            and self._room_grid_cache_matches(
                grid_info,
                scene_data,
                confidence_data,
            )
        )
        if previous_grid is not None and not cache_hit:
            self._room_split_shrink_allowed.update(
                self._rooms_with_transferred_cells(previous_grid, grid_info, scene_data))
        self.last_room_grid_cache_hit = cache_hit
        if not cache_hit or room_merges:
            self._room_grid_epoch += 1
            self._portal_room_ids_cache.clear()
        if cache_hit:
            # Retain the store-owned immutable copies from the previous call;
            # this avoids two more full-grid list allocations on the hot path.
            stored_scene_data = previous_grid["scene_data"]
            stored_confidence_data = previous_grid["confidence_data"]
        else:
            stored_scene_data = self._owned_room_grid_values(scene_data)
            stored_confidence_data = self._owned_room_grid_values(confidence_data)
        self.room_grid = {
            "info": grid_info,
            "scene_data": stored_scene_data,
            "confidence_data": stored_confidence_data,
        }
        self._apply_room_merges(room_merges or {})
        redirects_before = self._room_redirect_signature()
        if cache_hit:
            # This is a new room-grid receipt (same cell content, newer
            # source/header), so it still counts as one room observation for
            # the temporal room admission gate.
            self._refresh_room_nodes_from_cached_stats(count_observation=True)
        else:
            self._refresh_room_nodes_from_grid()
        # A force-stable room refresh can legitimately publish a grid where a
        # previously named room has disappeared before the temporal merge
        # confirmation is emitted.  Do not leave that room selectable merely
        # because no formal redirect has been committed yet.  Existing child
        # nodes are re-grounded against this latest grid below when possible.
        self._retire_absent_room_nodes_from_current_grid()
        self._rebuild_relations()
        # Portal-child resolution can introduce a redirect during relation
        # rebuild.  Invalidate the cache so the next frame re-aggregates with
        # the new resolved room IDs instead of reusing stale statistics.
        if redirects_before != self._room_redirect_signature():
            self._room_grid_stats_cache = None
        self._bump_revision()

    def set_room_geometries(self, rooms):
        self._room_grid_stats_cache = None
        self.room_geometries = {}
        for room in rooms or []:
            room_id = room.get("room_id")
            if room_id is None:
                continue
            room_id = int(room_id)
            self.room_geometries[room_id] = {
                "center": self._grounded_center(room.get("center") or room.get("aabb_center") or [float(room_id), 0.0, 0.0]),
                "aabb_center": self._grounded_center(room.get("aabb_center") or room.get("center") or [float(room_id), 0.0, 0.0]),
                "aabb_size": self._room_box_size(room.get("aabb_size") or [0.5, 0.5, self.room_box_height]),
                "name": str(room.get("name") or room_node_label(self.room_id_to_name, room_id)),
                "cell_count": int(room.get("cell_count", 0)),
            }
            self.room_id_to_name.setdefault(room_id, self.room_geometries[room_id]["name"])
            node = self._ensure_room_node(room_id)
            geom = self.room_geometries[room_id]
            stable_center, stable_size = self._accept_room_geometry(
                room_id,
                geom["aabb_center"],
                geom["aabb_size"],
                self.room_geometry_stability_frames,
            )
            node.name = geom["name"]
            # Room summaries are refreshed alongside graph revisions.  They may
            # describe a smaller or slightly shifted frontier envelope, but the
            # public box is the stable, monotonic geometry owned by the room ID.
            node.centroid = list(stable_center)
            node.aabb_center = list(stable_center)
            node.aabb_size = list(stable_size)
            node.attributes["cell_count"] = geom["cell_count"]

    def update_observations(
        self, observations, stamp=None, source_mode=None, capture_step=None
    ):
        now = float(stamp if stamp is not None else time.time())
        if source_mode:
            self.source_mode = str(source_mode)
        if capture_step is not None:
            self.capture_step = int(capture_step)
        self._rebuild_identity_index()
        if self.source_mode in {"realtime_gt_observation", "detector_online"}:
            for node in self.nodes.values():
                if node.type not in {"scene", "room"}:
                    node.attributes["_was_visible_previous_update"] = bool(
                        node.is_currently_visible
                    )
                    node.is_currently_visible = False
        for raw_observation in observations:
            observation = normalize_observation(raw_observation)
            node = self._find_or_create_node(observation)
            self._apply_observation(node, observation, now)
            self._index_node_identity(node)
        # ObjectMapStore can legitimately hand off a physical doorway to a
        # new detector track after an occlusion or a large viewpoint change.
        # The opaque track id is not a semantic identity, so collapse portal
        # nodes again at the graph boundary before rebuilding room relations.
        self._merge_duplicate_portal_nodes()
        # Portal deduplication can remove or move graph nodes.  No further
        # observation association happens in this batch, so reconcile lazily
        # at the next batch boundary instead of rebuilding O(N) here.
        self._invalidate_spatial_index()
        for node in self.nodes.values():
            node.attributes.pop("_was_visible_previous_update", None)
        # Room topology/statistics are produced by update_room_grid(), which
        # already owns the expensive full-grid scan and keeps a cache for
        # unchanged labels. Detector observations arrive at 10 Hz; rescanning
        # a multi-million-cell room grid here would hold the mapper lock for
        # hundreds of milliseconds on every YOLO receipt. Reapply the cached
        # per-room statistics instead, with the helper's full-scan fallback
        # for startup/legacy callers that have no cache yet.
        # Detector frames reuse the last room-grid statistics.  They must not
        # be counted as additional room observations: otherwise a 10 Hz YOLO
        # stream can satisfy the two-room-frame gate before a second room grid
        # has actually arrived.
        self._refresh_room_nodes_from_cached_stats(count_observation=False)
        self._rebuild_relations(now=now)
        self._refresh_missing_room_nodes_from_observations()
        self._bump_revision()

    def update_interaction_result(self, result, stamp=None):
        node_id = str(result.get("node_id") or "")
        instance_id = str(
            result.get("object_id") or result.get("instance_id") or ""
        )
        # An evaluator object-skill result has already resolved its opaque
        # object ID against the evaluator's current public frame.  Prefer that
        # public identity over a navigation-side node ID, which can be stale
        # when a direct bbox drawer scan was routed to a different graph node.
        prefer_instance_id = bool(instance_id) and str(
            result.get("source") or ""
        ).casefold() == "evaluator_object_skill"
        node = None
        if prefer_instance_id:
            node = next(
                (
                    candidate
                    for candidate in self.nodes.values()
                    if self._node_matches_identity(candidate, instance_id)
                ),
                None,
            )
        if node is None:
            node = self.nodes.get(node_id)
        if node is None and instance_id:
            node = next(
                (
                    candidate
                    for candidate in self.nodes.values()
                    if self._node_matches_identity(candidate, instance_id)
                ),
                None,
            )
        if node is None:
            source_object_name = str(result.get("source_object_name") or "")
            if source_object_name:
                node = next(
                    (
                        candidate
                        for candidate in self.nodes.values()
                        if self._node_matches_identity(candidate, source_object_name)
                    ),
                    None,
                )
        if node is None:
            return False
        # M1 can finish an interaction without issuing a force command when it
        # sees an already-open doorway.  Preserve that terminal observation as
        # an authoritative unavailable+open portal result; otherwise the
        # scheduler recreates the same interaction candidate on every graph
        # revision.  The aperture evidence is retained for the OCC bridge.
        result = dict(result)
        observation_only_open = (
            node.type == "portal"
            and bool(result.get("success"))
            and str(result.get("observation_outcome") or "").strip().casefold()
            == "finish_without_action"
            and str(result.get("state") or result.get("post_state") or "")
            .strip()
            .casefold()
            in {"open", "opened", "ajar", "static_open"}
        )
        if observation_only_open:
            result.setdefault("action", "open")
            aperture = result.get("portal_aperture_evidence")
            # M1 alone cannot upgrade topology.  Keep an unavailable terminal
            # until the occupancy/room lane has observed connectivity on both
            # sides; the later attribute patch then promotes it to static_open.
            result["state"] = (
                "static_open"
                if _portal_has_visual_and_map_open_evidence(node, aperture)
                else "unavailable"
            )
            result.setdefault("interaction_capability", "unavailable")
            result.setdefault("interactable", False)
            result.setdefault("source", "mllm_observation_finish_without_action")
        now = float(stamp if stamp is not None else time.time())
        pre_state = str(node.interaction.get("state", "unknown"))
        resolved_state, inferred_from_action = _resolved_interaction_state(result)
        # The resolver intentionally discards failed postconditions; retain
        # the rejected claim in gate diagnostics without applying it.
        requested_state = resolved_state
        if requested_state is None and result.get("success") is False:
            requested_state = result.get("state") or result.get("post_state")
        result_state_allowed, result_state_gate_reason = _portal_result_state_gate(
            node, result, requested_state
        )
        if node.type == "portal" and str(requested_state or "").casefold() in {
            "open",
            "opened",
            "ajar",
            "static_open",
        }:
            node.attributes["portal_result_state_gate"] = {
                "accepted": bool(result_state_allowed),
                "requested_state": str(requested_state or "").casefold(),
                "reason": result_state_gate_reason,
                "event_id": str(result.get("event_id") or ""),
            }
        if not result_state_allowed:
            resolved_state = None
            inferred_from_action = False
        observed_step = result.get(
            "capture_step",
            result.get("step", result.get("result_published_step")),
        )
        try:
            observed_step = int(observed_step) if observed_step is not None else None
        except (TypeError, ValueError):
            observed_step = None
        sequence_type = str(result.get("sequence_type") or "").strip().casefold()
        # The evaluator's public drawer-scan result historically omitted
        # ``sequence_type`` and exposed only the verification source.  Treat
        # that source as the same completed scan contract so a successful
        # closed-after-scan result remains terminal in the graph and is not
        # regenerated as another container interaction.
        if (
            not sequence_type
            and str(result.get("verification_source") or "").strip().casefold()
            == "drawer_scan_backend"
            and node.type == "container"
        ):
            sequence_type = "drawer_scan"
        if bool(result.get("success")) and sequence_type == "drawer_scan":
            grounded_regions = [
                str(item.get("region_id") or "")
                for item in list(result.get("grounded_regions") or [])
                if isinstance(item, dict) and str(item.get("region_id") or "")
            ]
            node.interaction.update(
                {
                    "drawer_scan_completed": True,
                    "drawer_scan_completed_step": observed_step,
                    "drawer_scan_completed_event_id": str(result.get("event_id") or ""),
                    "drawer_scan_covered_region_ids": grounded_regions,
                    "drawer_scan_covered_region_count": len(grounded_regions),
                }
            )
        result_source = str(
            result.get("source")
            or result.get("verification_source")
            or "interaction_result"
        )
        if resolved_state is not None:
            node.interaction["state"] = str(resolved_state)
            node.interaction["state_source"] = (
                "successful_action_postcondition"
                if inferred_from_action
                else result_source
            )
            node.interaction["state_confidence"] = float(
                result.get("confidence", 1.0)
            )
            node.interaction["state_observed_step"] = observed_step
            node.interaction["state_evidence"] = (
                f"action:{str(result.get('action') or 'unknown').casefold()}:success"
                if inferred_from_action
                else str(result.get("evidence") or result_source)
            )
        resolved_capability = _interaction_result_capability(result)
        static_capability = resolved_capability == "static"
        blocked_capability = resolved_capability == "blocked"
        unavailable_capability = resolved_capability == "unavailable"
        if (
            resolved_state is not None
            and result.get("success") is True
            and not static_capability
            and not blocked_capability
            and not unavailable_capability
        ):
            # A successful physical action is the rule lane's only evidence
            # that the portal is actually operable.  Keep this separate from
            # the visual MLLM capability assertion.
            node.interaction.update(
                {
                    "is_interactable": True,
                    "interaction_mode": str(
                        node.interaction.get("interaction_mode") or "open_close"
                    ),
                    "capability": "confirmed",
                    "capability_source": "executor_feedback",
                    "capability_confidence": float(result.get("confidence", 1.0)),
                    "capability_observed_step": observed_step,
                    "capability_evidence": "successful_interaction_feedback",
                }
            )
        elif static_capability:
            node.interaction.update(
                {
                    "is_interactable": False,
                    "interaction_mode": "none",
                    "state": str(resolved_state or "static"),
                    "state_source": str(
                        result.get("source")
                        or result.get("verification_source")
                        or "interaction_capability_check"
                    ),
                    "state_confidence": float(result.get("confidence", 1.0)),
                    "state_observed_step": observed_step,
                    "state_evidence": "interaction_capability_feedback",
                    "capability": "static",
                    "capability_source": "executor_feedback",
                    "capability_confidence": float(result.get("confidence", 1.0)),
                    "capability_observed_step": observed_step,
                    "capability_evidence": "interaction_capability_feedback",
                    "failure_reason": str(
                        result.get("reason") or "non_articulated"
                    ),
                }
            )
        elif blocked_capability or unavailable_capability:
            terminal_state = "blocked"
            terminal_state_source = str(
                result.get("source")
                or result.get("verification_source")
                or "interaction_capability_check"
            )
            terminal_state_evidence = "interaction_capability_feedback"
            if unavailable_capability:
                # Capability and aperture state are orthogonal.  A fixed
                # doorway may be permanently open or permanently closed; do
                # not erase that physical distinction by storing the generic
                # capability token as its state.  Only trusted pre-action or
                # already-gated graph state is used here, so a failed action
                # still cannot manufacture an open passage.
                graph_pre_state = str(pre_state or "unknown").strip().casefold()
                # Never promote aperture state from a failed result's
                # ``pre_state`` payload.  Open is admitted only from an already
                # gated graph state, M1+OCC agreement below, or a separately
                # successful force result handled by the success lane.
                observed_pre_state = graph_pre_state
                observation_only_open_confirmed = bool(
                    observation_only_open
                    and result_state_allowed
                    and str(resolved_state or "").casefold() == "static_open"
                )
                if observation_only_open_confirmed:
                    # The M1 observation itself is the authoritative aperture
                    # result for a non-articulated/already-open portal.  Keep
                    # executor capability unavailable while exposing the
                    # orthogonal static-open topology state.
                    terminal_state = "static_open"
                    terminal_state_source = "mllm_observation_finish_without_action"
                    terminal_state_evidence = "m1_open_aperture_observation"
                visual_map_open = bool(
                    node.type == "portal"
                    and _portal_has_visual_and_map_open_evidence(
                        node,
                        node.attributes.get("portal_aperture_evidence"),
                    )
                )
                if visual_map_open:
                    # A failed non-articulated action only establishes executor
                    # capability.  Keep aperture state orthogonal: M1-visible
                    # open space plus independently observed OCC connectivity
                    # is sufficient to classify this as unavailable/open even
                    # when the target touches an image edge.
                    terminal_state = "static_open"
                    terminal_state_source = "mllm_aperture+occupancy_connectivity"
                    terminal_state_evidence = "m1_open_aperture_and_occ_connectivity"
                elif observation_only_open_confirmed:
                    pass
                elif observed_pre_state in {"open", "opened", "ajar", "static_open"}:
                    terminal_state = "static_open"
                elif observed_pre_state in {"closed", "static_closed"}:
                    terminal_state = "static_closed"
                else:
                    terminal_state = "unavailable"
                node.attributes["portal_unavailable_state_resolution"] = {
                    "state": terminal_state,
                    "m1_open_aperture": bool(
                        isinstance(
                            node.attributes.get("portal_aperture_evidence"), dict
                        )
                        and node.attributes["portal_aperture_evidence"].get(
                            "open_aperture"
                        )
                        == "visible"
                    ),
                    "observed_open_connectivity": bool(
                        _portal_has_observed_open_connectivity(node)
                    ),
                    "reason": (
                        "m1_open_aperture_and_occ_connectivity"
                        if visual_map_open
                        else "preserved_preinteraction_state"
                    ),
                    "event_id": str(result.get("event_id") or ""),
                }
            node.interaction.update(
                {
                    "is_interactable": False,
                    "interaction_mode": "none",
                    "state": terminal_state,
                    "state_source": terminal_state_source,
                    "state_confidence": float(result.get("confidence", 1.0)),
                    "state_observed_step": observed_step,
                    "state_evidence": terminal_state_evidence,
                    "capability": resolved_capability,
                    "capability_source": "executor_feedback",
                    "capability_confidence": float(result.get("confidence", 1.0)),
                    "capability_observed_step": observed_step,
                    "capability_evidence": "interaction_capability_feedback",
                    "failure_reason": str(
                        result.get("reason")
                        or result.get("failure_reason")
                        or "interaction_unavailable"
                    ),
                }
            )
        self._refresh_planner_state_fields(node)
        state = node.interaction.get("state", "unknown")

        history = list(node.interaction.get("operation_history") or [])
        event_id = str(result.get("event_id") or f"interaction_{self.interaction_event_counter:06d}")
        if not any(entry.get("event_id") == event_id for entry in history):
            self.interaction_event_counter += 1
            history_entry = {
                    "event_id": event_id,
                    "action": str(result.get("action") or result.get("interaction_mode") or "unknown"),
                    "timestamp": now,
                    "pre_state": pre_state,
                    "post_state": str(state),
                    "success": bool(result.get("success", True)),
                    "execution_cost": float(result.get("execution_cost", result.get("cost", 1.0))),
                    "verification_source": str(
                        "successful_action_postcondition"
                        if inferred_from_action
                        else result.get("verification_source")
                        or result.get("source")
                        or "interaction_result"
                    ),
                }
            if result.get("interaction_group_id"):
                history_entry["interaction_group_id"] = str(
                    result["interaction_group_id"]
                )
            approach_goal = list(result.get("approach_goal_xyyaw") or [])
            if len(approach_goal) >= 2:
                history_entry["approach_goal_xyyaw"] = [
                    float(value) for value in approach_goal[:3]
                ]
            history.append(history_entry)
        node.interaction["operation_history"] = history
        self._update_interaction_group_memory(node, result)
        if (
            resolved_state is not None
            or static_capability
            or blocked_capability
            or unavailable_capability
        ):
            node.attributes["interaction_state_override"] = {
                key: node.interaction.get(key)
                for key in (
                    "state",
                    "state_source",
                    "state_confidence",
                    "state_observed_step",
                    "state_evidence",
                    "traversable",
                    "requires_interaction",
                    "is_interactable",
                    "interaction_mode",
                    "capability",
                    "capability_source",
                    "capability_confidence",
                    "capability_observed_step",
                    "capability_evidence",
                    "failure_reason",
                )
                if key in node.interaction
            }
            node.attributes["interaction_state_override"]["event_id"] = event_id
            node.attributes["interaction_state_override"]["timestamp"] = now
        node.last_seen = now
        if (
            self.portal_child_room_enabled
            and node.type == "portal"
            and _is_confirmed_portal_open_result(result, state)
        ):
            self._ensure_open_portal_child_room(node, history)
        self._rebuild_relations(now=now)
        self._bump_revision()
        return True

    def _update_interaction_group_memory(self, node, result):
        sequence_results = list(result.get("interaction_group_results") or [])
        if sequence_results:
            for group_result in sequence_results:
                self._update_interaction_group_memory(node, dict(group_result or {}))
            return
        interaction = node.interaction
        group_id = str(
            result.get("interaction_group_id") or result.get("region_id") or ""
        )
        if not group_id:
            return
        success = bool(result.get("success", True))
        completed = {
            str(value)
            for value in interaction.get("completed_interaction_groups") or []
        }
        failed = {
            str(value) for value in interaction.get("failed_interaction_groups") or []
        }
        if success:
            completed.add(group_id)
            failed.discard(group_id)
        else:
            failed.add(group_id)
        interaction["completed_interaction_groups"] = sorted(completed)
        interaction["failed_interaction_groups"] = sorted(failed)

    @staticmethod
    def _refresh_planner_state_fields(node):
        interaction = node.interaction
        state = str(interaction.get("state") or "unknown")
        if node.type == "portal":
            interaction["traversable"] = (
                True
                if state in {"open", "ajar", "static_open"}
                else False
                if state in {"closed", "blocked", "static_closed", "unavailable"}
                else None
            )
            interaction["requires_interaction"] = bool(
                interaction.get("is_interactable")
                and state
                not in {
                    "open",
                    "ajar",
                    "static_open",
                    "blocked",
                    "static_closed",
                    "unavailable",
                }
            )

            return
        interaction["traversable"] = (
            True
            if state in {"open", "ajar", "static_open"}
            else False
            if state in {"closed", "blocked", "static_closed", "unavailable"}
            else None
        )
        interaction["requires_interaction"] = bool(
            interaction.get("is_interactable") and state in {"closed", "unknown"}
        )

    @staticmethod
    def _expanded_room_geometry(accepted_center, accepted_size, center, size):
        old_min_x = float(accepted_center[0]) - 0.5 * float(accepted_size[0])
        old_max_x = float(accepted_center[0]) + 0.5 * float(accepted_size[0])
        old_min_y = float(accepted_center[1]) - 0.5 * float(accepted_size[1])
        old_max_y = float(accepted_center[1]) + 0.5 * float(accepted_size[1])
        new_min_x = float(center[0]) - 0.5 * float(size[0])
        new_max_x = float(center[0]) + 0.5 * float(size[0])
        new_min_y = float(center[1]) - 0.5 * float(size[1])
        new_max_y = float(center[1]) + 0.5 * float(size[1])
        if (
            new_min_x >= old_min_x - 1e-6
            and new_max_x <= old_max_x + 1e-6
            and new_min_y >= old_min_y - 1e-6
            and new_max_y <= old_max_y + 1e-6
        ):
            return list(accepted_center), list(accepted_size)
        min_x, max_x = min(old_min_x, new_min_x), max(old_max_x, new_max_x)
        min_y, max_y = min(old_min_y, new_min_y), max(old_max_y, new_max_y)
        return (
            [0.5 * (min_x + max_x), 0.5 * (min_y + max_y), float(center[2])],
            [max(0.1, max_x - min_x), max(0.1, max_y - min_y), float(size[2])],
        )

    def apply_attribute_patch(self, patch, stamp=None):
        object_id = str(patch.get("object_id") or "")
        node = self.nodes.get(object_id)
        if node is None:
            node = next(
                (
                    candidate
                    for candidate in self.nodes.values()
                    if self._node_matches_identity(candidate, object_id)
                ),
                None,
            )
        if node is None:
            return False
        attribute_status = str(patch.get("attribute_status") or "ready")
        confidence = 0.0
        if attribute_status not in {"pending", "failed", "stale"}:
            try:
                confidence = float(patch.get("confidence", 0.0))
            except (TypeError, ValueError, OverflowError):
                return False
            # Rejected evidence must not change an accepted name/state or
            # overwrite the confirmation metadata used by portal topology.
            if not math.isfinite(confidence) or not 0.5 <= confidence <= 1.0:
                return False
        patch_stamp = float(stamp if stamp is not None else time.time())
        request_sequence = int(patch.get("request_sequence", 0) or 0)
        latest_request_sequence = int(
            node.attributes.get("attribute_request_sequence", 0) or 0
        )
        if request_sequence and latest_request_sequence and request_sequence < latest_request_sequence:
            return False
        # A Module-1 call is asynchronous: later ordinary scans do not make
        # the requested visual evidence invalid.  Request sequence plus the
        # evidence signature are the freshness contract; capture step remains
        # telemetry for diagnosing inference lag only.
        patch_signature = str(patch.get("observation_signature") or "")
        latest_request_signature = str(
            node.attributes.get("attribute_request_signature") or ""
        )
        if (
            request_sequence
            and latest_request_sequence
            and request_sequence == latest_request_sequence
            and latest_request_signature
            and patch_signature
            and patch_signature != latest_request_signature
        ):
            return False
        patch_frame_index = patch.get(
            "observation_frame_index", patch.get("observation_capture_step")
        )
        current_frame_index = node.attributes.get("last_observation_frame_index")
        try:
            patch_frame_index = int(patch_frame_index)
            current_frame_index = int(current_frame_index)
        except (TypeError, ValueError):
            patch_frame_index = None
            current_frame_index = None
        observation_lag_steps = (
            max(0, current_frame_index - patch_frame_index)
            if patch_frame_index is not None and current_frame_index is not None
            else None
        )
        request_signature = latest_request_signature
        if patch_signature and (
            attribute_status == "pending" or not request_signature
        ):
            request_signature = patch_signature
        if str(node.attributes.get("attribute_status") or "").casefold() == "ready":
            # Also preserve ready evidence from older serialized graphs that
            # predate attribute_last_ready before overwriting request status.
            node.attributes["attribute_last_ready"] = {
                key: node.attributes.get(key) for key in _SEMANTIC_CONFIRMATION_FIELDS
            }
        node.attributes.update(
            {
                "attribute_status": attribute_status,
                "attribute_request_sequence": max(latest_request_sequence, request_sequence),
                "attribute_response_lag_sec": float(
                    patch.get("response_lag_sec", 0.0) or 0.0
                ),
                "attribute_queue_lag_sec": float(patch.get("queue_lag_sec", 0.0) or 0.0),
                "attribute_total_lag_sec": float(patch.get("total_lag_sec", 0.0) or 0.0),
                "attribute_error": str(patch.get("error") or ""),
                "attribute_observation_stamp_sec": float(
                    patch.get("observation_stamp_sec", patch_stamp) or patch_stamp
                ),
                "attribute_observation_frame_index": patch_frame_index,
                "attribute_observation_signature": str(
                    patch.get("observation_signature") or ""
                ),
                "attribute_request_signature": request_signature,
                "attribute_observation_lag_steps": observation_lag_steps,
            }
        )
        if attribute_status in {"pending", "failed", "stale"}:
            self._bump_revision()
            return True
        verified_state_override = dict(
            node.attributes.get("interaction_state_override") or {}
        )
        has_verified_interaction_state = bool(
            verified_state_override.get("event_id")
        )
        interaction_class = normalize_label(patch.get("interaction_class"))
        patch_source = str(patch.get("source") or "mllm_attribute_inference")
        is_visual_mllm_patch = "mllm" in patch_source.casefold()
        m1_observed_name = _m1_observed_object_name(patch)
        # A generic detector label (locker/safe/object) is only a hypothesis
        # until M1 has independently confirmed the refrigerator name twice.
        # One crop is not enough to distinguish the field refrigerator from a
        # drinking-water dispenser, which is exactly the failure mode seen on
        # the physical platform.  Count distinct observation signatures rather
        # than the number of image tiles in one request; repeated tiles from a
        # single frame are not independent evidence.
        source_labels = {
            normalize_label(value)
            for value in (
                node.attributes.get("source_semantic_name"),
                node.attributes.get("source_category"),
            )
            if normalize_label(value)
        }
        source_is_refrigerator = any(
            _is_refrigerator_label(value) for value in source_labels
        )
        m1_recheck_required = bool(
            source_labels.intersection(
                {"locker", "safe", "water_dispenser", "dispenser"}
            )
        )
        m1_refrigerator_answer = bool(
            is_visual_mllm_patch
            and confidence >= 0.5
            and _is_refrigerator_label(m1_observed_name)
        )
        confirmation_signatures = [
            str(value)
            for value in list(
                node.attributes.get("m1_refrigerator_confirmation_signatures") or []
            )
            if str(value)
        ]
        # The compact view signature intentionally stays stable while the
        # robot holds still.  For this *recheck* it must be combined with the
        # actual capture frame, otherwise two M1 calls on later frames would be
        # mistaken for one piece of evidence forever.
        confirmation_signature = str(patch_signature or "")
        if patch_frame_index is not None:
            confirmation_signature += f"|frame:{patch_frame_index}"
        elif not confirmation_signature:
            confirmation_signature = (
                f"capture:{patch.get('observation_capture_step')}"
            )
        confirmation_count = int(
            node.attributes.get("m1_refrigerator_confirmation_count", 0) or 0
        )
        if m1_refrigerator_answer and m1_recheck_required and not source_is_refrigerator:
            if confirmation_signature and confirmation_signature not in confirmation_signatures:
                confirmation_signatures.append(confirmation_signature)
                confirmation_signatures = confirmation_signatures[-8:]
                confirmation_count = len(confirmation_signatures)
        m1_refrigerator_confirmed = bool(
            source_is_refrigerator or confirmation_count >= 2
        )
        m1_refrigerator_pending = bool(
            m1_refrigerator_answer
            and m1_recheck_required
            and not m1_refrigerator_confirmed
            and not source_is_refrigerator
        )
        source_observation_is_portal = bool(
            source_labels.intersection({"portal", "door", "doorway", "gate"})
        )
        portal_morphology = _public_portal_morphology(patch)
        portal_aperture_evidence = _public_portal_aperture_evidence(patch)
        portal_visual_evidence = bool(
            isinstance(portal_morphology, dict)
            and str(portal_morphology.get("door_leaf") or "")
            in {"present", "absent"}
            and float(portal_morphology.get("confidence", 0.0) or 0.0) >= 0.70
        ) or bool(
            isinstance(portal_aperture_evidence, dict)
            and str(portal_aperture_evidence.get("open_aperture") or "")
            == "visible"
            and float(portal_aperture_evidence.get("confidence", 0.0) or 0.0)
            >= 0.70
        )
        portal_name_is_door = bool(
            not m1_observed_name or m1_observed_name in PORTAL_LABELS
        )
        portal_identity_authorized = bool(
            interaction_class == "portal"
            and portal_name_is_door
            and (source_observation_is_portal or portal_visual_evidence)
            and _portal_geometry_plausible(node.aabb_size)
        )
        # M1 is the visual authority for both the concrete name and the
        # interaction class.  A sufficiently confident M1 answer may correct
        # a YOLO-established container/portal type; lower-confidence answers
        # retain the existing topology but are still recorded for diagnostics.
        # The physical detector lane uses M1 as the final visual refinement
        # authority.  Restricted-GT/replay lanes still need the established
        # topology hysteresis: a single asynchronous crop must not turn a
        # known portal into a container (or vice versa) while the source
        # observation stream is authoritative.  Keeping this distinction at
        # the store boundary preserves the physical fix without regressing
        # simulator/replay graph semantics.
        m1_class_override = bool(
            is_visual_mllm_patch
            and self.source_mode == "detector_online"
            and confidence >= 0.5
            and interaction_class in {"portal", "container", "support", "object"}
            and (not m1_refrigerator_answer or m1_refrigerator_confirmed)
            and (interaction_class != "portal" or portal_identity_authorized)
        )
        m1_noninteractive_override = bool(
            is_visual_mllm_patch
            and confidence >= 0.5
            and (
                (
                    not bool(patch.get("interactable", False))
                    and interaction_class in {"none", "unknown"}
                )
                or (
                    interaction_class == "portal"
                    and not portal_identity_authorized
                )
            )
        )
        portal_state_consensus_accepted = bool(
            patch.get("portal_state_consensus_accepted", True)
        )
        requested_patch_state = str(
            patch.get("coarse_state") or "unknown"
        ).strip().casefold()
        observed_topology_type = str(
            node.attributes.get("topology_type")
            or node.attributes.get("observation_node_type")
            or ""
        ).strip().casefold()
        # M1 classifies a visual crop, while a portal changes map topology.
        # In particular, never let one delayed M1 response promote a
        # source-observed container/support/object to a portal: that formerly
        # let an opened fridge inherit a pending doorway clear.  Non-topology
        # refinements (for example generic object -> container) remain allowed.
        portal_promotion_rejected = bool(
            interaction_class == "portal"
            and (
                (
                    observed_topology_type
                    and observed_topology_type != "portal"
                    and not m1_class_override
                )
                or not portal_identity_authorized
            )
        )
        (
            container_type_locked,
            container_state_rejected,
            container_hysteresis_reason,
        ) = _container_m1_hysteresis(
            node,
            interaction_class,
            requested_patch_state,
            bool(patch.get("interactable", False)),
            is_visual_mllm_patch=is_visual_mllm_patch,
        )
        portal_type_locked, portal_hysteresis_reason = _portal_m1_type_hysteresis(
            node,
            interaction_class,
            is_visual_mllm_patch=is_visual_mllm_patch,
        )
        if m1_class_override:
            # The old hysteresis protects YOLO/source topology from a delayed
            # one-frame M1 flip.  For physical navigation the user explicitly
            # makes M1 authoritative, so a confident visual class is allowed
            # to replace that hypothesis.
            container_type_locked = False
            container_state_rejected = False
            portal_type_locked = False
        if (
            not has_verified_interaction_state
            and confidence >= 0.5
            and interaction_class in {"portal", "container", "support", "object"}
            and not portal_promotion_rejected
            and not container_type_locked
            and not portal_type_locked
            and not m1_refrigerator_pending
        ):
            node.type = interaction_class
        if (
            m1_noninteractive_override
            and not has_verified_interaction_state
            and not container_type_locked
            and not portal_type_locked
        ):
            # A detector may call a wall or furniture a fridge/door.  Once M1
            # explicitly says it is not interactive, keep the node in the
            # semantic map but remove it from the interaction-candidate pool.
            node.type = "object"
        if (
            m1_observed_name
            and is_visual_mllm_patch
            and confidence >= 0.5
            and not m1_refrigerator_pending
        ):
            # Keep the stable track/instance ID and source detector name, but
            # replace the public semantic name used by candidates, graph views,
            # and subsequent M2 prompts with M1's visual result.
            node.label = m1_observed_name
            node.name = m1_observed_name
            node.attributes.update(
                {
                    "semantic_name": m1_observed_name,
                    "category": m1_observed_name,
                    "m1_observed_object_name": m1_observed_name,
                    "m1_name_override": True,
                    "m1_name_confidence": confidence,
                }
            )
            if _is_refrigerator_label(m1_observed_name):
                # A generic detector crop may be promoted to a refrigerator
                # by M1.  Capture its body geometry before a later source
                # frame can overwrite the mutable public type/name.
                _remember_m1_refrigerator_reference(
                    node, m1_observed_name, confidence
                )
        parts = list(patch.get("interaction_parts") or [])
        for deprecated_key in (
            "interaction_groups",
            "interaction_group_source",
            "joint_infos",
            "observation_evidence",
        ):
            node.attributes.pop(deprecated_key, None)
        node.attributes.update(
            {
                "attribute_source": str(patch.get("source") or "mllm"),
                "attribute_model": str(patch.get("model_name") or ""),
                "attribute_confidence": confidence,
                "mllm_interaction_class": interaction_class,
                "m1_observed_object_name": m1_observed_name,
                "m1_detector_class_hypothesis": str(
                    patch.get("m1_detector_class_hypothesis") or ""
                ),
                "m1_class_override": m1_class_override,
                "m1_refrigerator_confirmation_count": confirmation_count,
                "m1_refrigerator_confirmation_signatures": confirmation_signatures,
                "m1_refrigerator_confirmed": m1_refrigerator_confirmed,
                "m1_refrigerator_pending_confirmation": m1_refrigerator_pending,
                "m1_pending_observed_object_name": (
                    m1_observed_name if m1_refrigerator_pending else ""
                ),
                "m1_noninteractive_override": m1_noninteractive_override,
                "mllm_portal_promotion_rejected": portal_promotion_rejected,
                "evidence_frame_ids": list(patch.get("evidence_frame_ids") or []),
                "affordances": list(patch.get("affordances") or []),
                "interaction_parts": parts,
                "mllm_interaction_parts": parts,
                "portal_state_consensus": dict(
                    patch.get("portal_state_consensus") or {}
                ),
                "portal_state_consensus_accepted": portal_state_consensus_accepted,
            }
        )
        if is_visual_mllm_patch and (
            node.type == "container"
            or observed_topology_type == "container"
        ):
            # Keep this as public, bounded diagnostic metadata.  It makes a
            # rejected one-frame answer visible to replay/inspection without
            # allowing the answer to alter topology or interaction capability.
            node.attributes["mllm_container_type_hysteresis"] = {
                "locked_type": "container",
                "requested_type": interaction_class or "unknown",
                "requested_state": requested_patch_state or "unknown",
                "accepted": not bool(container_type_locked),
                "state_accepted": not bool(container_state_rejected),
                "reason": container_hysteresis_reason,
                "observation_capture_step": patch_frame_index,
            }
            node.attributes["mllm_container_type_locked"] = True
            node.attributes["mllm_container_state_rejected"] = bool(
                container_state_rejected
            )
        if is_visual_mllm_patch and (
            node.type == "portal" or observed_topology_type == "portal"
        ):
            node.attributes["mllm_portal_type_hysteresis"] = {
                "locked_type": "portal",
                "requested_type": interaction_class or "unknown",
                "accepted": not bool(portal_type_locked),
                "reason": portal_hysteresis_reason,
                "observation_capture_step": patch_frame_index,
            }
            node.attributes["mllm_portal_type_locked"] = True
        # M1 owns the pre-interaction visual state. Persist its compact public
        # contract so candidate generation can derive an approach from the
        # observed view rather than from a simulator/oracle orientation.
        for key in (
            "view_state",
            "view_state_confidence",
            "front_surface_visible",
            "front_surface_confidence",
            "approach_ready",
            "needs_reobserve",
            "visual_evidence_truncated",
            "visual_evidence_truncated_edges",
        ):
            if key in patch:
                node.attributes[key] = patch.get(key)
        if any(
            key in patch
            for key in (
                "view_state",
                "front_surface_visible",
                "approach_ready",
                "needs_reobserve",
            )
        ):
            node.attributes["view_state_source"] = str(
                patch.get("source") or "mllm_attribute_inference"
            )
            node.attributes["approach_source"] = node.attributes["view_state_source"]
            node.attributes["attribute_is_current"] = True
        if container_state_rejected or portal_type_locked:
            # The crop is still useful as a reason to re-observe, but its
            # contradictory class/state must not make a fridge or drawer
            # disappear from the interaction graph for this frame.
            node.attributes["needs_reobserve"] = True
            node.attributes["approach_ready"] = False
            node.attributes["attribute_is_current"] = False
        for key in (
            "targeted_refresh",
            "targeted_refresh_request_id",
            "targeted_refresh_sequence",
            "targeted_refresh_reason",
            "targeted_refresh_minimum_capture_step",
            "targeted_refresh_image_sequence",
            "evidence_frame_ids",
            "evidence_capture_steps",
            "evidence_observation_pose_xyyaw",
            "m1_evidence_image_count",
            "target_bbox_containment",
        ):
            if key in patch:
                node.attributes[key] = patch.get(key)
        portal_morphology = _public_portal_morphology(patch)
        portal_aperture_evidence = _public_portal_aperture_evidence(patch)
        if node.type == "portal" and portal_morphology is not None:
            node.attributes["portal_morphology"] = portal_morphology
        if node.type == "portal" and portal_aperture_evidence is not None:
            node.attributes["portal_aperture_evidence"] = portal_aperture_evidence
        elif node.type != "portal":
            node.attributes.pop("portal_morphology", None)
            node.attributes.pop("portal_aperture_evidence", None)
        previous_history = list(node.interaction.get("operation_history") or [])
        latest_operation_stamp = max(
            [
                float(item.get("timestamp", 0.0) or 0.0)
                for item in previous_history
                if isinstance(item, dict)
            ],
            default=0.0,
        )
        patch_state = str(
            patch.get("coarse_state")
            or node.interaction.get("state")
            or "unknown"
        )
        unavailable_open_reconciliation = bool(
            has_verified_interaction_state
            and is_visual_mllm_patch
            and node.type == "portal"
            and portal_state_consensus_accepted
            and str(verified_state_override.get("capability") or "").casefold()
            == "unavailable"
            and patch_state.casefold() in {"open", "ajar", "static_open"}
            and _portal_has_visual_and_map_open_evidence(
                node,
                portal_aperture_evidence,
            )
        )
        state_was_updated = (
            not has_verified_interaction_state
            and latest_operation_stamp <= patch_stamp
            and is_visual_mllm_patch
            and (node.type != "portal" or portal_state_consensus_accepted)
            and not container_state_rejected
            and not portal_type_locked
        ) or unavailable_open_reconciliation
        if (
            is_visual_mllm_patch
            and node.type == "portal"
            and not portal_state_consensus_accepted
        ):
            node.attributes["portal_state_gate"] = {
                "accepted": False,
                "requested_state": str(patch_state).casefold(),
                "reason": "portal_state_consensus_pending",
                "observation_capture_step": patch_frame_index,
            }
        if state_was_updated:
            if unavailable_open_reconciliation:
                patch_state = "static_open"
                portal_state_allowed = True
                portal_state_gate_reason = (
                    "unavailable_capability_m1_open_and_map_confirmed"
                )
                node.attributes["portal_unavailable_state_resolution"] = {
                    "state": "static_open",
                    "m1_open_aperture": True,
                    "observed_open_connectivity": True,
                    "reason": "m1_open_aperture_and_occ_connectivity",
                    "event_id": str(verified_state_override.get("event_id") or ""),
                }
            else:
                portal_state_allowed, portal_state_gate_reason = _portal_visual_state_gate(
                    node,
                    patch_state,
                    portal_morphology,
                    portal_aperture_evidence,
                    visual_evidence_truncated=bool(
                        patch.get("visual_evidence_truncated", False)
                    ),
                )
            if node.type == "portal":
                node.attributes["portal_state_gate"] = {
                    "accepted": bool(portal_state_allowed),
                    "requested_state": str(patch_state).casefold(),
                    "reason": portal_state_gate_reason,
                    "observation_capture_step": patch_frame_index,
                }
            if not portal_state_allowed:
                # Preserve the preceding closed/unknown state rather than
                # turning a weak MLLM "open" into a route-opening claim.
                # The candidate layer will then request a real, one-shot
                # interaction whose public feedback owns the terminal state.
                state_was_updated = False
        if state_was_updated:
            patch_capability = str(
                patch.get("interaction_capability")
                or patch.get("capability")
                or ""
            ).strip().casefold()
            patch_interactable = bool(patch.get("interactable", False))
            if patch_state.casefold() == "blocked" or patch_capability in {
                "blocked",
                "unsupported",
                "unavailable",
                "locked",
            }:
                patch_capability = "blocked"
                patch_interactable = False
            elif patch_state.casefold() == "static_open":
                patch_capability = "static"
                patch_interactable = False
            elif patch_capability not in {"static", "unknown"}:
                patch_capability = "unknown"
            node.interaction.update(
                {
                    "is_interactable": patch_interactable,
                    "interaction_mode": (
                        str(node.interaction.get("interaction_mode") or "none")
                        if patch_interactable
                        else "none"
                    ),
                    "state": patch_state,
                    "state_source": patch_source,
                    "state_confidence": confidence,
                    "state_observed_step": patch_frame_index,
                    "state_evidence": "mllm_visual_observation",
                    "capability": patch_capability,
                    "capability_source": patch_source,
                    "capability_confidence": confidence,
                    "capability_observed_step": patch_frame_index,
                    "capability_evidence": "mllm_visual_observation",
                    "failure_reason": str(
                        patch.get("failure_reason")
                        or patch.get("reason")
                        or ""
                    ),
                    "expected_effect": str(
                        patch.get("expected_effect")
                        or (
                            "unlock_connectivity"
                            if node.type == "portal"
                            else "reveal_contents"
                        )
                    ),
                    "operation_history": previous_history,
                }
            )
            if unavailable_open_reconciliation:
                # Preserve executor capability while accepting the independent
                # visual+map aperture state.  Capability and aperture are two
                # separate axes exposed as unavailable/open in visualization.
                for key in (
                    "is_interactable",
                    "interaction_mode",
                    "capability",
                    "capability_source",
                    "capability_confidence",
                    "capability_observed_step",
                    "capability_evidence",
                    "failure_reason",
                ):
                    if key in verified_state_override:
                        node.interaction[key] = verified_state_override[key]
                node.interaction["state_source"] = (
                    "mllm_aperture+occupancy_connectivity"
                )
                node.interaction["state_evidence"] = (
                    "m1_open_aperture_and_occ_connectivity"
                )
        if state_was_updated:
            self._refresh_planner_state_fields(node)
            node.attributes["interaction_state_override"] = {
                key: node.interaction.get(key)
                for key in (
                    "state",
                    "state_source",
                    "state_confidence",
                    "state_observed_step",
                    "state_evidence",
                    "traversable",
                    "requires_interaction",
                    "is_interactable",
                    "interaction_mode",
                    "capability",
                    "capability_source",
                    "capability_confidence",
                    "capability_observed_step",
                    "capability_evidence",
                    "failure_reason",
                    "expected_effect",
                )
                if key in node.interaction
            }
            node.attributes["interaction_state_override"]["timestamp"] = patch_stamp
            if unavailable_open_reconciliation and verified_state_override.get(
                "event_id"
            ):
                node.attributes["interaction_state_override"]["event_id"] = str(
                    verified_state_override["event_id"]
                )

        # Keep the interaction mode canonical even when a repeated M1 patch
        # does not change the closed/open state.  Otherwise the first patch
        # may leave mode=none and every later patch preserves that value.
        if (
            bool(patch.get("interactable", False))
            and _is_refrigerator_label(m1_observed_name or node.label)
            and any(
                str(part.get("type") or "").casefold() in {"door", "lid", "drawer"}
                for part in (patch.get("interaction_parts") or [])
                if isinstance(part, dict)
            )
        ):
            node.interaction["interaction_mode"] = "open_close"
            override = dict(node.attributes.get("interaction_state_override") or {})
            override["interaction_mode"] = "open_close"
            node.attributes["interaction_state_override"] = override

        node.attributes["attribute_updated_at"] = patch_stamp
        node.attributes["attribute_last_ready"] = {
            key: node.attributes.get(key) for key in _SEMANTIC_CONFIRMATION_FIELDS
        }
        self._update_persistent_semantic_gate(node)
        # This very response may complete the two-frame + M1 gate. Evaluate
        # topology afterwards so the first confirmed open observation need
        # not wait for another model call to create its graph-only far side.
        if (state_was_updated and self.portal_child_room_enabled
            and node.type == "portal"
            and str(node.interaction.get("state") or "").casefold() in {"open", "ajar"}):
            self._ensure_open_portal_child_room(node, previous_history)
        self._rebuild_relations(now=patch_stamp)
        self._bump_revision()
        return True

    def apply_room_attribute_patch(self, patch, stamp=None):
        """Apply an asynchronous room-only Module-1 result.

        Object state patches and room evidence patches deliberately have
        independent IDs, queues, and capture-step versions.  This prevents a
        delayed room classification from changing an interaction object and
        lets the rule-based room prior continue to serve as a fallback.
        """

        try:
            room_id = int(patch.get("room_id"))
        except (TypeError, ValueError):
            return False
        requested_node_id = str(patch.get("room_node_id") or "")
        room_node = self.nodes.get(requested_node_id)
        if room_node is None or room_node.type != "room":
            room_node = next(
                (
                    candidate
                    for candidate in self.nodes.values()
                    if candidate.type == "room"
                    and candidate.room_id is not None
                    and int(candidate.room_id) == room_id
                ),
                None,
            )
        if (
            room_node is None
            or not room_node.attributes.get("active", True)
            or room_node.attributes.get("is_potential_room", False)
        ):
            return False

        attribute_status = str(
            patch.get("room_attribute_status")
            or patch.get("attribute_status")
            or "ready"
        ).casefold()
        patch_stamp = float(stamp if stamp is not None else time.time())
        request_sequence = int(patch.get("request_sequence", 0) or 0)
        latest_request_sequence = int(
            room_node.attributes.get("room_attribute_request_sequence", 0) or 0
        )
        if (
            request_sequence
            and latest_request_sequence
            and request_sequence < latest_request_sequence
        ):
            return False

        # As for object attributes, a later map scan only changes the age of
        # the text evidence.  A room result is current when it targets the
        # current active room and matches the latest requested evidence
        # signature; capture step is explicitly retained as telemetry.
        patch_signature = str(patch.get("observation_signature") or "")
        latest_request_signature = str(
            room_node.attributes.get("room_attribute_request_signature") or ""
        )
        if (
            request_sequence
            and latest_request_sequence
            and request_sequence == latest_request_sequence
            and latest_request_signature
            and patch_signature
            and patch_signature != latest_request_signature
        ):
            return False

        patch_capture_step = patch.get(
            "observation_capture_step", patch.get("capture_step")
        )
        current_capture_step = self.capture_step
        try:
            patch_capture_step = int(patch_capture_step)
        except (TypeError, ValueError):
            patch_capture_step = None
        try:
            current_capture_step = int(current_capture_step)
        except (TypeError, ValueError):
            current_capture_step = None
        observation_lag_steps = (
            max(0, current_capture_step - patch_capture_step)
            if patch_capture_step is not None and current_capture_step is not None
            else None
        )
        request_signature = latest_request_signature
        if patch_signature and (
            attribute_status == "pending" or not request_signature
        ):
            request_signature = patch_signature

        room_node.attributes.update(
            {
                "room_attribute_status": attribute_status,
                "room_attribute_request_sequence": max(
                    latest_request_sequence, request_sequence
                ),
                "room_attribute_observation_stamp_sec": float(
                    patch.get("observation_stamp_sec", patch_stamp) or patch_stamp
                ),
                "room_attribute_observation_capture_step": patch_capture_step,
                "room_attribute_observation_signature": str(
                    patch.get("observation_signature") or ""
                ),
                "room_attribute_request_signature": request_signature,
                "room_attribute_observation_lag_steps": observation_lag_steps,
                "room_attribute_queue_lag_sec": float(
                    patch.get("queue_lag_sec", 0.0) or 0.0
                ),
                "room_attribute_response_lag_sec": float(
                    patch.get("response_lag_sec", 0.0) or 0.0
                ),
                "room_attribute_total_lag_sec": float(
                    patch.get("total_lag_sec", 0.0) or 0.0
                ),
                "room_attribute_error": str(patch.get("error") or ""),
                "room_attribute_fallback": bool(patch.get("fallback", False)),
            }
        )
        if attribute_status == "ready":
            room_node.attributes.update(
                {
                    "room_mllm_attribute": normalize_label(
                        patch.get("room_attribute") or "unknown"
                    )
                    or "unknown",
                    "room_mllm_attribute_confidence": max(
                        0.0, min(1.0, float(patch.get("confidence", 0.0) or 0.0))
                    ),
                    "room_mllm_attribute_evidence_object_ids": [
                        str(item)
                        for item in patch.get("evidence_object_ids") or []
                        if str(item)
                    ],
                    "room_mllm_attribute_source": str(
                        patch.get("source") or "mllm_room_attribute_inference"
                    ),
                    "room_mllm_attribute_model": str(patch.get("model_name") or ""),
                    "room_mllm_attribute_updated_at": patch_stamp,
                }
            )
        self._refresh_room_attributes()
        self._bump_revision()
        return True

    def as_graph_bundle(self, stamp=None):
        now = float(stamp if stamp is not None else time.time())
        for node in self.nodes.values():
            node.state_age_sec = max(0.0, now - float(node.last_seen)) if node.last_seen is not None else 0.0
            node.graph_revision = self.graph_revision
        nodes = sorted(self.nodes.values(), key=lambda item: item.id)
        edges = sorted(self.edges.values(), key=lambda item: item.id)
        semantic_node_ids = [node.id for node in nodes]
        semantic_edge_ids = [edge.id for edge in edges]

        interaction_core = {node.id for node in nodes if node.type in {"portal", "support", "container"}}
        interaction_edge_ids = []
        for edge in edges:
            if edge.src_id in interaction_core or edge.dst_id in interaction_core:
                interaction_core.add(edge.src_id)
                interaction_core.add(edge.dst_id)
                interaction_edge_ids.append(edge.id)

        navigation_hints = self._build_navigation_hints()
        navigation_node_ids = sorted({hint.node_id for hint in navigation_hints})
        navigation_edge_ids = [
            edge.id
            for edge in edges
            if edge.src_id in navigation_node_ids or edge.dst_id in navigation_node_ids
        ]

        return SceneGraphBundle(
            scene_id=self.scene_id,
            episode_id=self.episode_id,
            source_mode=self.source_mode,
            graph_revision=self.graph_revision,
            timestamp=now,
            capture_step=self.capture_step,
            nodes=nodes,
            edges=edges,
            semantic_node_ids=semantic_node_ids,
            semantic_edge_ids=semantic_edge_ids,
            interaction_node_ids=sorted(interaction_core),
            interaction_edge_ids=sorted(set(interaction_edge_ids)),
            navigation_node_ids=navigation_node_ids,
            navigation_edge_ids=navigation_edge_ids,
            navigation_hints=navigation_hints,
        )

    def as_graph_dict(self, stamp=None):
        return self.as_graph_bundle(stamp=stamp).to_dict()

    def as_navigation_hints(self):
        return [hint.to_dict() for hint in self._build_navigation_hints()]

    def prune_stale_nodes(self, stale_after_sec, now=None):
        stale_after_sec = float(stale_after_sec)
        if stale_after_sec <= 0.0:
            return
        now = float(now if now is not None else time.time())
        stale_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if (
                node.type != "room"
                and node.last_seen is not None
                and now - float(node.last_seen) > stale_after_sec
                and not _has_persistent_semantic_evidence(node)
                # Successful interaction state is mission memory, not a
                # detector track. Keep it even when the object leaves view;
                # ordinary unconfirmed/stale detections may be reclaimed.
                and not bool(node.interaction.get("operation_history"))
                and str(node.interaction.get("state") or "").casefold()
                not in {"open", "opened", "ajar", "static_open", "blocked", "unavailable"}
            )
        ]
        if not stale_ids:
            return
        stale_id_set = set(stale_ids)
        for node_id in stale_ids:
            self.nodes.pop(node_id, None)
        self._invalidate_spatial_index()
        self.edges = {
            edge_id: edge
            for edge_id, edge in self.edges.items()
            if edge.src_id not in stale_id_set and edge.dst_id not in stale_id_set
        }
        self._rebuild_relations(now=now)

    def _find_or_create_node(self, observation):
        instance_id = str(observation.get("instance_id") or "")
        private_instance_id = str(observation.get("private_instance_id") or "")
        identities = tuple(
            identity
            for identity in (instance_id, private_instance_id)
            if identity
        )
        if identities:
            if self._identity_index is None:
                self._rebuild_identity_index()
            for identity in identities:
                node_id = self._identity_index.get(identity)
                node = self.nodes.get(node_id) if node_id is not None else None
                if node is not None and self._node_matches_identity(node, identity):
                    return node
            # The index is rebuilt at every observation-batch boundary and is
            # updated whenever a node is observed/created.  A miss is therefore
            # definitive for normal ROS callers; avoid falling back to an
            # O(N) scan for every newly created track in a large first batch.

        node_type = infer_node_type(observation)
        label = normalize_label(observation.get("semantic_name"))
        best = None
        best_dist = None
        candidates = self._spatial_candidates(observation, node_type, label)
        if candidates is None:
            # Compatibility for a minimal test double or a legacy store that
            # bypassed ``__init__``. Production batches build the index before
            # reaching this branch.
            candidates = self.nodes.values()
        for node in candidates:
            if node.type in {"scene", "room"}:
                continue
            if node.type != node_type:
                continue
            # M1 and detector paths may alternate the public label between
            # ``door`` and ``portal`` while retaining the same physical
            # surface. Treat portal-family labels as equivalent for spatial
            # association so a label canonicalization cannot split a track.
            portal_equivalent = (
                node_type == "portal"
                and node.label in _PORTAL_MATCH_LABELS
                and label in _PORTAL_MATCH_LABELS
            )
            if node.label != label and not portal_equivalent:
                continue
            dist = distance_xy(node.centroid, observation["position"])
            if dist <= self.match_distance and (best is None or dist < best_dist):
                best = node
                best_dist = dist
        if best is not None:
            return best

        node_id = self._make_node_id(node_type, observation)
        node = SceneGraphNode(
            id=node_id,
            type=node_type,
            label=label or node_type,
            name=str(observation.get("name") or label or node_type),
        )
        self.nodes[node_id] = node
        # The caller indexes after ``_apply_observation`` has populated the
        # instance ID, centroid, and final type.  Indexing this empty shell
        # here would immediately be repeated and adds one spatial-bucket
        # update per newly created detector track.
        return node

    @staticmethod
    def _portal_axis_delta(first_yaw, second_yaw):
        """Return the unoriented (pi-periodic) angle between door axes."""

        try:
            delta = abs(float(first_yaw) - float(second_yaw))
        except (TypeError, ValueError):
            return math.inf
        return abs((delta + 0.5 * math.pi) % math.pi - 0.5 * math.pi)

    def _portal_nodes_cross_view_match(self, first, second):
        """Whether two graph portal nodes describe one physical doorway."""

        if first is None or second is None:
            return False
        if first.type != "portal" or second.type != "portal":
            return False
        first_attrs = first.attributes or {}
        second_attrs = second.attributes or {}
        first_yaw = first_attrs.get("interaction_reference_yaw", first_attrs.get("yaw"))
        second_yaw = second_attrs.get("interaction_reference_yaw", second_attrs.get("yaw"))
        if first_yaw is None or second_yaw is None:
            return distance_xy(first.aabb_center, second.aabb_center) <= max(
                self.match_distance, 0.5
            )
        # Close-range partial masks can rotate the PCA axis by roughly one
        # radian even when the observed doorway is unchanged.  The spatial
        # normal/tangent/height gates below remain authoritative for rejecting
        # adjacent doors.
        if self._portal_axis_delta(first_yaw, second_yaw) > 1.20:
            return False

        first_center = list(first.aabb_center or first.centroid or [0.0, 0.0, 0.0])
        second_center = list(second.aabb_center or second.centroid or [0.0, 0.0, 0.0])
        first_size = list(first.aabb_size or [0.0, 0.0, 0.0])
        second_size = list(second.aabb_size or [0.0, 0.0, 0.0])
        # PCA/OBB yaw can jump substantially for a thin, partially visible
        # door leaf.  If the two axis-aligned RGB-D boxes are nevertheless
        # almost coincident, that is stronger duplicate evidence than the
        # unstable yaw.  The minimum-volume overlap keeps adjacent doors
        # separate even when their centers are nearby.
        center_distance = distance_xy(first_center, second_center)
        if center_distance <= 0.35:
            overlap_volume = 1.0
            for axis in range(3):
                first_half = 0.5 * abs(float(first_size[axis]))
                second_half = 0.5 * abs(float(second_size[axis]))
                overlap_axis = max(
                    0.0,
                    first_half + second_half
                    - abs(float(second_center[axis]) - float(first_center[axis])),
                )
                overlap_volume *= overlap_axis
            first_volume = math.prod(max(abs(float(value)), 1e-3) for value in first_size[:3])
            second_volume = math.prod(max(abs(float(value)), 1e-3) for value in second_size[:3])
            overlap_ratio = overlap_volume / max(min(first_volume, second_volume), 1e-6)
            if overlap_ratio >= 0.15:
                return True
        tangent_x = math.cos(float(first_yaw))
        tangent_y = math.sin(float(first_yaw))
        normal_x = -tangent_y
        normal_y = tangent_x
        delta_x = float(second_center[0]) - float(first_center[0])
        delta_y = float(second_center[1]) - float(first_center[1])
        if abs(delta_x * normal_x + delta_y * normal_y) > 0.45:
            return False

        tangent_distance = abs(delta_x * tangent_x + delta_y * tangent_y)
        first_width = max(abs(float(first_size[0])), abs(float(first_size[1])), 1e-3)
        second_width = max(abs(float(second_size[0])), abs(float(second_size[1])), 1e-3)
        tangent_overlap = max(
            0.0, 0.5 * (first_width + second_width) - tangent_distance
        )
        if tangent_overlap / max(min(first_width, second_width), 1e-3) < 0.20:
            return False

        first_height = max(abs(float(first_size[2])), 1e-3)
        second_height = max(abs(float(second_size[2])), 1e-3)
        vertical_distance = abs(float(second_center[2]) - float(first_center[2]))
        vertical_overlap = max(
            0.0, 0.5 * (first_height + second_height) - vertical_distance
        )
        return (
            vertical_overlap / max(min(first_height, second_height), 1e-3)
            >= 0.35
        )

    def _merge_duplicate_portal_nodes(self):
        """Merge split doorway graph nodes while retaining the strongest node."""

        portals = [node for node in self.nodes.values() if node.type == "portal"]
        if len(portals) < 2:
            return
        ordered = sorted(
            portals,
            key=lambda node: (
                # A detector handoff is not a new semantic doorway. Keep the
                # confirmed public ID (including an in-flight M1 refresh), so
                # room ownership and pending results remain attached to it.
                -int(_has_m1_portal_confirmation(node.attributes)),
                -int(node.observation_count or 0),
                -(float(node.confidence or 0.0)),
                str(node.id),
            ),
        )
        removed = set()
        aliases = {}
        for index, keeper in enumerate(ordered):
            if keeper.id in removed:
                continue
            for duplicate in ordered[index + 1 :]:
                if duplicate.id in removed or not self._portal_nodes_cross_view_match(
                    keeper, duplicate
                ):
                    continue
                # Confirmed evidence already ranks first. Do not copy a
                # duplicate's bare "ready" status: it may lack valid evidence
                # and must not overwrite a confirmed node's pending refresh.
                keeper_attrs = keeper.attributes
                duplicate_attrs = duplicate.attributes
                keeper.observation_count = max(
                    int(keeper.observation_count or 0),
                    int(duplicate.observation_count or 0),
                )
                keeper.confidence = max(
                    float(keeper.confidence or 0.0),
                    float(duplicate.confidence or 0.0),
                )
                keeper.is_currently_visible = bool(
                    keeper.is_currently_visible or duplicate.is_currently_visible
                )
                self._update_persistent_semantic_gate(keeper)
                keeper.last_seen = max(
                    value
                    for value in (keeper.last_seen, duplicate.last_seen)
                    if value is not None
                ) if keeper.last_seen is not None or duplicate.last_seen is not None else None
                connected = set(keeper_attrs.get("connected_room_ids") or [])
                connected.update(duplicate_attrs.get("connected_room_ids") or [])
                if connected:
                    keeper_attrs["connected_room_ids"] = sorted(connected)
                aliases[duplicate.id] = keeper.id
                removed.add(duplicate.id)

        if not removed:
            return
        for edge_id, edge in list(self.edges.items()):
            edge.src_id = aliases.get(edge.src_id, edge.src_id)
            edge.dst_id = aliases.get(edge.dst_id, edge.dst_id)
            if edge.src_id == edge.dst_id:
                self.edges.pop(edge_id, None)
        # Relations are rebuilt below from the surviving portal set. Remove
        # stale duplicate nodes now so they cannot re-enter candidates or M1.
        for node_id in removed:
            self.nodes.pop(node_id, None)

    @staticmethod
    def _node_matches_identity(node, identity: str) -> bool:
        """Match public or private routing IDs without exposing private IDs."""

        identity = str(identity or "")
        if not identity:
            return False
        attributes = node.attributes or {}
        return identity in {
            str(node.id or ""),
            str(attributes.get("instance_id") or ""),
            str(attributes.get("source_object_name") or ""),
            str(attributes.get("_private_instance_id") or ""),
            str(attributes.get("_private_source_object_name") or ""),
        }

    @staticmethod
    def _node_identity_values(node):
        attributes = node.attributes or {}
        return (
            str(node.id or ""),
            str(attributes.get("instance_id") or ""),
            str(attributes.get("source_object_name") or ""),
            str(attributes.get("_private_instance_id") or ""),
            str(attributes.get("_private_source_object_name") or ""),
        )

    @staticmethod
    def _spatial_label_key(node_type, label):
        """Return the label family used by the track-association index."""

        label = normalize_label(label)
        if node_type == "portal" and label in PORTAL_LABELS:
            # Portal aliases are intentionally equivalent in the historical
            # matcher; keep them in one bucket while retaining the exact
            # semantic check below.
            return "__portal__"
        return label

    def _spatial_bucket_key(self, node):
        """Return a coarse XY bucket for one non-room graph node."""

        if node is None or getattr(node, "type", None) in {"scene", "room"}:
            return None
        center = list(getattr(node, "centroid", None) or [])
        if len(center) < 2:
            return None
        try:
            cell_size = float(self._spatial_cell_size)
            x = float(center[0])
            y = float(center[1])
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (cell_size, x, y)):
            return None
        return (
            str(getattr(node, "type", "object") or "object"),
            self._spatial_label_key(
                getattr(node, "type", "object"),
                getattr(node, "label", ""),
            ),
            math.floor(x / cell_size),
            math.floor(y / cell_size),
        )

    def _observation_spatial_key(self, observation, node_type, label, dx=0, dy=0):
        try:
            position = observation.get("position") or []
            x = float(position[0])
            y = float(position[1])
            cell_size = float(self._spatial_cell_size)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (cell_size, x, y)):
            return None
        return (
            str(node_type or "object"),
            self._spatial_label_key(node_type, label),
            math.floor(x / cell_size) + int(dx),
            math.floor(y / cell_size) + int(dy),
        )

    def _rebuild_spatial_index(self):
        index = defaultdict(list)
        buckets = {}
        order = {}
        for insertion_order, node in enumerate(self.nodes.values()):
            node_id = str(getattr(node, "id", "") or "")
            if not node_id:
                continue
            order[node_id] = insertion_order
            key = self._spatial_bucket_key(node)
            if key is None:
                continue
            index[key].append(node_id)
            buckets[node_id] = key
        self._spatial_index = dict(index)
        self._spatial_node_bucket = buckets
        self._spatial_node_order = order

    def _invalidate_spatial_index(self):
        """Force a rebuild after nodes are removed/merged outside a batch."""

        self._spatial_index = None
        self._spatial_node_bucket = {}
        self._spatial_node_order = {}

    def _index_node_spatial(self, node):
        """Insert/update one node in the current batch's spatial index."""

        if self._spatial_index is None or node is None:
            return
        node_id = str(getattr(node, "id", "") or "")
        if not node_id or getattr(node, "type", None) in {"scene", "room"}:
            return
        old_key = self._spatial_node_bucket.get(node_id)
        new_key = self._spatial_bucket_key(node)
        if old_key == new_key:
            if new_key is not None:
                bucket = self._spatial_index.setdefault(new_key, [])
                if node_id not in bucket:
                    bucket.append(node_id)
            self._spatial_node_order.setdefault(
                node_id, len(self._spatial_node_order)
            )
            return
        if old_key is not None:
            old_bucket = self._spatial_index.get(old_key)
            if old_bucket is not None:
                try:
                    old_bucket.remove(node_id)
                except ValueError:
                    pass
                if not old_bucket:
                    self._spatial_index.pop(old_key, None)
            self._spatial_node_bucket.pop(node_id, None)
        self._spatial_node_order.setdefault(node_id, len(self._spatial_node_order))
        if new_key is not None:
            self._spatial_index.setdefault(new_key, []).append(node_id)
            self._spatial_node_bucket[node_id] = new_key

    def _spatial_candidates(self, observation, node_type, label):
        """Return nearby indexed nodes in historical insertion order."""

        if self._spatial_index is None:
            return None
        try:
            position = observation.get("position") or []
            x = float(position[0])
            y = float(position[1])
            cell_size = float(self._spatial_cell_size)
        except (AttributeError, IndexError, TypeError, ValueError):
            return []
        if not all(math.isfinite(value) for value in (cell_size, x, y)):
            return []
        base_x = math.floor(x / cell_size)
        base_y = math.floor(y / cell_size)
        prefix = (
            str(node_type or "object"),
            self._spatial_label_key(node_type, label),
        )
        candidate_ids = []
        seen = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bucket = self._spatial_index.get(
                    (prefix[0], prefix[1], base_x + dx, base_y + dy), ()
                )
                for node_id in bucket:
                    if node_id not in seen:
                        seen.add(node_id)
                        candidate_ids.append(node_id)
        if not candidate_ids:
            return []
        if len(candidate_ids) > 1:
            candidate_ids.sort(
                key=lambda node_id: self._spatial_node_order.get(node_id, 1 << 60)
            )
        return [
            self.nodes[node_id]
            for node_id in candidate_ids
            if node_id in self.nodes
        ]

    def _rebuild_identity_index(self):
        index = {}
        for node in self.nodes.values():
            for identity in self._node_identity_values(node):
                if identity:
                    # Preserve the first match, which is the same insertion
                    # order used by the historical full scan on collisions.
                    index.setdefault(identity, node.id)
        self._identity_index = index
        self._rebuild_spatial_index()

    def _index_node_identity(self, node):
        if self._identity_index is None:
            return
        for identity in self._node_identity_values(node):
            if identity:
                self._identity_index.setdefault(identity, node.id)
        self._index_node_spatial(node)

    def _ensure_scene_node(self):
        node_id = f"scene_{sanitize_token(self.episode_id or self.scene_id)}"
        existing = next((node for node in self.nodes.values() if node.type == "scene"), None)
        if existing is not None:
            if existing.id != node_id:
                self.nodes.pop(existing.id, None)
                existing.id = node_id
                self.nodes[node_id] = existing
            existing.name = self.episode_id or self.scene_id
            existing.label = normalize_label(self.scene_id) or "scene"
            existing.attributes.update(
                {
                    "scene_id": self.scene_id,
                    "episode_id": self.episode_id,
                    "source_mode": self.source_mode,
                }
            )
            return existing
        node = SceneGraphNode(
            id=node_id,
            type="scene",
            label=normalize_label(self.scene_id) or "scene",
            name=self.episode_id or self.scene_id,
            confidence=1.0,
            is_currently_visible=True,
            attributes={
                "scene_id": self.scene_id,
                "episode_id": self.episode_id,
                "source_mode": self.source_mode,
            },
        )
        self.nodes[node_id] = node
        self._index_node_identity(node)
        return node

    def _make_node_id(self, node_type, observation):
        instance_id = sanitize_token(observation.get("instance_id") or "")
        if instance_id:
            # A restricted-GT door reference is already the complete public
            # identity.  Keep the graph node aligned with it instead of
            # emitting ``portal_door_0001`` and making every downstream
            # consumer translate between two public names.
            if (
                node_type == "portal"
                and bool(observation.get("minimal_gt_observation"))
                and is_public_door_id(instance_id)
            ):
                if instance_id not in self.nodes:
                    return instance_id
            node_id = f"{node_type}_{instance_id}"
            if node_id not in self.nodes:
                return node_id
        label = sanitize_token(observation.get("semantic_name") or node_type)
        while True:
            node_id = f"{node_type}_{label}_{self.next_node_index}"
            self.next_node_index += 1
            if node_id not in self.nodes:
                return node_id

    def _apply_observation(self, node, observation, now):
        minimal_gt = bool(observation.get("minimal_gt_observation"))
        interaction_state_override = dict(
            node.attributes.get("interaction_state_override") or {}
        )
        observed_node_type = infer_node_type(observation)
        if observed_node_type == "portal" and not _portal_geometry_plausible(
            observation.get("aabb_size")
        ):
            # A detector portal label with a degenerate RGB-D box is usually a
            # clipped mask/depth failure, not a doorway. Keep it as an
            # ordinary object until M1 supplies valid topology evidence.
            observed_node_type = "object"
        node.type = observed_node_type
        node.label = normalize_label(observation.get("semantic_name")) or node.type
        node.name = str(observation.get("name") or node.label or node.type)
        # Physical detector frames refresh geometry, not an accepted M1
        # identity. Keep source labels below for diagnostics; replay/GT
        # observations remain authoritative in their own lanes.
        retained_m1_name = ""
        if self.source_mode == "detector_online":
            if node.attributes.get("m1_name_override"):
                retained_m1_name = normalize_label(
                    node.attributes.get("semantic_name")
                )
                if retained_m1_name:
                    node.label = retained_m1_name
                    node.name = retained_m1_name
            if node.attributes.get("m1_class_override"):
                node.type = node.attributes.get("mllm_interaction_class") or node.type
            elif node.attributes.get("m1_noninteractive_override"):
                node.type = "object"
        if minimal_gt and node.type == "portal":
            node.label = "door"
            node.name = "door"
        observed_aabb_center = list(observation["aabb_center"])
        observed_aabb_size = list(observation["aabb_size"])
        aabb_reused_previous = False
        if (
            not self._valid_aabb_size(observed_aabb_size)
            and self._valid_aabb_size(node.aabb_size)
        ):
            # A transient publisher/geometry failure must not erase a valid
            # portal or container footprint and force candidate generation to
            # fall back to a robot-bearing direction.
            observed_aabb_center = list(node.aabb_center)
            observed_aabb_size = list(node.aabb_size)
            aabb_reused_previous = True
        node.centroid = self._ground_non_room_centroid(
            observation["position"], observed_aabb_size
        )
        node.aabb_center = self._ground_non_room_centroid(
            observed_aabb_center, observed_aabb_size
        )
        node.aabb_size = observed_aabb_size
        node.room_id = observation.get("room_id") if observation.get("room_id") is not None else node.room_id
        node.confidence = max(float(node.confidence), float(observation.get("confidence", 0.0)))
        node.observation_count += 1
        if node.first_seen is None:
            node.first_seen = now
        node.last_seen = now
        node.is_currently_visible = True
        max_visible_pixels = max(
            int(node.attributes.get("max_visible_pixels", 0) or 0),
            int(observation.get("visible_pixels", 0) or 0),
        )
        max_visible_fraction = max(
            float(node.attributes.get("max_visible_fraction", 0.0) or 0.0),
            float(observation.get("visible_fraction", 0.0) or 0.0),
        )
        consecutive_observations = int(
            observation.get("consecutive_observations", 0) or 0
        )
        if minimal_gt:
            consecutive_observations = (
                int(node.attributes.get("consecutive_observations", 0) or 0) + 1
                if bool(node.attributes.get("_was_visible_previous_update"))
                else 1
            )
        max_consecutive_observations = max(
            int(node.attributes.get("max_consecutive_observations", 0) or 0),
            consecutive_observations,
        )

        observation_attributes = {
                # Source observations, rather than an asynchronous M1 class
                # proposal, own the topology class.  The public graph still
                # records M1's proposed class separately for diagnostics and
                # non-topological container refinement.
                "observation_node_type": observed_node_type,
                "topology_type": observed_node_type,
                "topology_type_source": "source_observation",
                # Keep the detector/source ontology separate from the mutable
                # M1-facing ``semantic_name``/``category`` fields below.  M1
                # can temporarily answer "refrigerator" for a bottle (or
                # promote a crop to ``container``); relation inference must
                # still be able to tell which labels came from the geometry
                # observation when the next graph rebuild runs.
                "source_semantic_name": normalize_label(
                    observation.get("semantic_name")
                ),
                "source_category": normalize_label(observation.get("category")),
                "instance_id": observation.get("instance_id") or node.attributes.get("instance_id") or "",
                "category": observation.get("category"),
                "candidate_labels": list(observation.get("candidate_labels") or []),
                "label_votes": dict(observation.get("label_votes") or {}),
                "connected_room_ids": list(observation.get("connected_room_ids") or []),
                # Keep direct room endpoints separate from the derived public
                # connectivity.  The latter may additionally contain a
                # post-open portal-child room while its free space is still
                # unobserved.
                "observed_connected_room_ids": list(
                    observation.get("connected_room_ids") or []
                ),
                "source": observation.get("source"),
                "visible_pixels": int(observation.get("visible_pixels", 0)),
                "max_visible_pixels": max_visible_pixels,
                "visible_fraction": float(
                    observation.get("visible_fraction", 0.0) or 0.0
                ),
                "max_visible_fraction": max_visible_fraction,
                "bbox_2d": list(observation.get("bbox_2d") or []),
                "consecutive_observations": consecutive_observations,
                "max_consecutive_observations": max_consecutive_observations,
                "tracking_confirmed": observation.get("tracking_confirmed"),
                "graph_admission_source": str(
                    observation.get("graph_admission_source") or ""
                ),
                "camera_name": observation.get("camera_name"),
                "frame_index": int(observation.get("frame_index", 0)),
                "last_observation_frame_index": int(
                    observation.get("frame_index", 0) or 0
                ),
                "episode_id": observation.get("episode_id"),
                "box_3d_frame_id": observation.get("box_3d_frame_id"),
                "viz_aabb_center": list(observation.get("viz_aabb_center") or observed_aabb_center),
                "viz_aabb_size": list(observation.get("viz_aabb_size") or observed_aabb_size),
                # Physical YOLOE publishes the oriented 3-D box in the
                # visualization fields while ``aabb_size`` remains a world
                # axis-aligned envelope. Preserve the oriented extent for
                # interaction-face geometry; otherwise a rotated appliance's
                # surface intersection is computed with the wrong half-widths.
                "interaction_reference_obb_size": list(
                    observation.get("viz_aabb_size") or observed_aabb_size
                ),
                "aabb_valid": bool(self._valid_aabb_size(observed_aabb_size)),
                "aabb_reused_previous": bool(aabb_reused_previous),
            }
        if not (minimal_gt and node.type == "portal"):
            observation_attributes["source_object_name"] = observation.get(
                "source_object_name"
            )
        else:
            # A restricted-GT portal has an opaque public ``instance_id``.
            # Never serialize its simulator source/body name as an attribute.
            observation_attributes.pop("source_object_name", None)
        private_instance_id = str(observation.get("private_instance_id") or "")
        private_source_object_name = str(
            observation.get("private_source_object_name") or ""
        )
        if private_instance_id:
            observation_attributes["_private_instance_id"] = private_instance_id
        if private_source_object_name:
            observation_attributes[
                "_private_source_object_name"
            ] = private_source_object_name
        interaction_axis = list(
            observation.get("interaction_approach_axis_xy") or []
        )
        if interaction_axis:
            observation_attributes["interaction_approach_axis_xy"] = interaction_axis
            interaction_axis_source = str(
                observation.get("interaction_approach_axis_source") or ""
            )
            if interaction_axis_source:
                observation_attributes[
                    "interaction_approach_axis_source"
                ] = interaction_axis_source
        if not minimal_gt:
            observation_attributes.update(
                {
                    "asset_id": observation.get("asset_id"),
                    "object_id": observation.get("object_id"),
                    "orientation": list(observation.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
                    "projected_bbox_2d": list(
                        observation.get("projected_bbox_2d") or []
                    ),
                }
            )
            if observation.get("yaw") is not None:
                observation_attributes["yaw"] = float(observation["yaw"])
        node.attributes.update(observation_attributes)
        if retained_m1_name:
            node.attributes["category"] = retained_m1_name
        # A new RGB/detection observation can make the previous M1 visual
        # judgment stale. Keep the judgment for diagnostics, but advertise it
        # as current only for a small causal capture window.
        attribute_capture_step = node.attributes.get(
            "attribute_observation_frame_index"
        )
        try:
            attribute_capture_step = int(attribute_capture_step)
            current_capture_step = int(observation.get("frame_index", 0) or 0)
            attribute_is_current = (
                str(node.attributes.get("attribute_status") or "").casefold()
                == "ready"
                and current_capture_step >= attribute_capture_step
                and current_capture_step - attribute_capture_step <= 2
            )
        except (TypeError, ValueError):
            attribute_is_current = False
        node.attributes["attribute_is_current"] = bool(attribute_is_current)
        source_object_name = str(
            observation.get("private_source_object_name")
            or observation.get("source_object_name")
            or observation.get("instance_id")
            or node.name
            or ""
        )
        geometry_override = self.interaction_geometry_overrides.get(
            source_object_name
        )
        if geometry_override:
            for key in (
                "interaction_approach_axis_xy",
                "interaction_approach_pose_xyyaw",
                "interaction_reference_aabb_center",
                "interaction_reference_aabb_size",
            ):
                values = list(geometry_override.get(key) or [])
                if values:
                    node.attributes[key] = values
            node.attributes["interaction_geometry_source"] = str(
                geometry_override.get("source") or "configured_override"
            )
        # Keep the first reliable refrigerator body box as an interaction
        # reference.  After the door opens, detectors frequently enlarge or
        # translate the live appliance box to include the door/contents; that
        # box is useful for visualisation but must not redefine the depth
        # corridor used to assign the newly visible items.  A configured
        # geometry override, when present, remains authoritative.
        refrigerator_label = next(
            (
                value
                for value in (
                    node.label,
                    observation.get("semantic_name"),
                    observation.get("category"),
                )
                if _is_refrigerator_label(value)
            ),
            None,
        )
        known_refrigerator_label = bool(
            refrigerator_label
            or _is_refrigerator_label(
                node.attributes.get("interaction_reference_label")
            )
        )
        if known_refrigerator_label:
            if not _is_refrigerator_label(
                node.attributes.get("interaction_reference_label")
            ):
                node.attributes["interaction_reference_label"] = normalize_label(
                    refrigerator_label
                )
            if not _valid_float_vector(
                node.attributes.get("interaction_reference_aabb_center"), 3
            ):
                node.attributes["interaction_reference_aabb_center"] = list(
                    observation["aabb_center"]
                )
            if not _valid_float_vector(
                node.attributes.get("interaction_reference_aabb_size"), 3
            ):
                node.attributes["interaction_reference_aabb_size"] = list(
                    observation["aabb_size"]
                )
            if (
                not _valid_float_scalar(
                    node.attributes.get("interaction_reference_yaw")
                )
                and observation.get("yaw") is not None
            ):
                try:
                    node.attributes["interaction_reference_yaw"] = float(
                        observation["yaw"]
                    )
                except (TypeError, ValueError):
                    pass
        for deprecated_key in (
            "parent",
            "children",
            "is_receptacle",
            "is_pickup_candidate",
            "is_articulable",
            "is_door",
            "is_movable_door",
            "joint_infos",
            "interaction_groups",
            "interaction_group_source",
            "observation_evidence",
        ):
            node.attributes.pop(deprecated_key, None)

        previous_interaction_memory = {
            key: node.interaction.get(key)
            for key in (
                "operation_history",
                "completed_interaction_groups",
                "failed_interaction_groups",
                "drawer_scan_completed",
                "drawer_scan_completed_step",
                "drawer_scan_completed_event_id",
                "drawer_scan_covered_region_ids",
                "drawer_scan_covered_region_count",
            )
            if key in node.interaction
        }
        node.interaction = default_interaction_payload(node.type, observation)
        if node.type == "portal":
            if not self._valid_aabb_size(
                node.attributes.get("interaction_reference_aabb_size")
            ):
                node.attributes["interaction_reference_aabb_center"] = list(
                    observed_aabb_center
                )
                node.attributes["interaction_reference_aabb_size"] = list(
                    observed_aabb_size
                )
            # Upgrade a legacy/reference portal yaw when a measured OBB yaw
            # arrives.  Older physical detections did not serialize yaw and
            # consequently left ``interaction_reference_yaw=0`` even when
            # the door's long edge was along Y.  Keep a valid measured yaw
            # stable, but replace that legacy axis-aligned value when the new
            # OBB clearly differs by ~90 degrees.
            observed_yaw = observation.get("yaw")
            if observed_yaw is not None:
                try:
                    observed_yaw = float(observed_yaw)
                except (TypeError, ValueError):
                    observed_yaw = None
            reference_yaw = node.attributes.get("interaction_reference_yaw")
            try:
                reference_yaw = (
                    None if reference_yaw is None else float(reference_yaw)
                )
            except (TypeError, ValueError):
                reference_yaw = None
            if observed_yaw is not None and math.isfinite(observed_yaw):
                if reference_yaw is None:
                    node.attributes["interaction_reference_yaw"] = observed_yaw
                else:
                    # Yaw is axial (theta and theta+pi are equivalent).  A
                    # mismatch above 45 degrees is not normal OBB jitter and
                    # identifies the old missing-yaw reference.
                    delta = abs(
                        math.atan2(
                            math.sin(observed_yaw - reference_yaw),
                            math.cos(observed_yaw - reference_yaw),
                        )
                    )
                    axial_delta = min(delta, abs(math.pi - delta))
                    if axial_delta > math.pi / 4.0:
                        node.attributes["interaction_reference_yaw"] = observed_yaw
        if interaction_state_override:
            for key in (
                "state",
                "state_source",
                "state_confidence",
                "state_observed_step",
                "state_evidence",
                "traversable",
                "requires_interaction",
                "is_interactable",
                "interaction_mode",
                "capability",
                "capability_source",
                "capability_confidence",
                "capability_observed_step",
                "capability_evidence",
                "expected_effect",
                "failure_reason",
            ):
                if key in interaction_state_override:
                    node.interaction[key] = interaction_state_override[key]

        node.interaction.update(previous_interaction_memory)

        self._update_persistent_semantic_gate(node)

    @staticmethod
    def _update_persistent_semantic_gate(node):
        """Latch persistence after two detector frames and a valid M1 result."""

        if node.type not in {"portal", "container"}:
            return
        action = persistence_action(node.type, node.observation_count, node.attributes)
        if action == "latch":
            node.attributes.update(
                {
                    "persistent_semantic_node": True,
                    "semantic_confirmation": "two_detector_frames_plus_m1",
                    "persistent_since": node.attributes.get("persistent_since")
                    or node.last_seen,
                }
            )
        elif action == "clear":
            # M1 may later reject a detector portal as a wall panel or other
            # non-door surface.  Clear the latch immediately so any synthetic
            # child room can be removed on the same callback.
            node.attributes.pop("persistent_semantic_node", None)
            node.attributes.pop("semantic_confirmation", None)
            node.attributes.pop("persistent_since", None)

    def _refresh_room_nodes_from_grid(self, geometry_stability_frames=None):
        if not self.room_grid:
            return
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        confidence_data = self.room_grid["confidence_data"]
        if grid_info is None or not scene_data:
            return
        if geometry_stability_frames is None:
            geometry_stability_frames = self.room_geometry_stability_frames
        # This path runs for every accepted room-grid revision. A typical
        # full-size occupancy grid contains roughly four million cells, so
        # materialising a Python tuple/list for every labelled cell makes the
        # graph update both slow and a long critical section in the mapper.
        # Aggregate the same world-space statistics in NumPy and only retain
        # per-room scalar statistics in Python.
        scene_values = np.asarray(scene_data, dtype=np.int64)
        valid_indices = np.flatnonzero(scene_values >= 0)
        if valid_indices.size == 0:
            self._cache_room_grid_statistics(grid_info, scene_data, confidence_data, [])
            return

        width = int(grid_info.width)
        resolution = float(grid_info.resolution)
        if resolution <= 0.0:
            raise ValueError("OccupancyGrid resolution must be positive")

        raw_room_ids, raw_inverse = np.unique(
            scene_values[valid_indices], return_inverse=True
        )
        resolved_by_raw_room_id = np.asarray(
            [self._resolve_room_id(room_id) for room_id in raw_room_ids],
            dtype=np.int64,
        )
        # Resolve/unique the small set of raw room labels first, then expand
        # the inverse map back to cells.  This is equivalent to uniquing the
        # per-cell resolved array but avoids a second full-grid sort.
        room_ids, raw_room_to_resolved_inverse = np.unique(
            resolved_by_raw_room_id,
            return_inverse=True,
        )
        room_inverse = raw_room_to_resolved_inverse[raw_inverse]

        mx = (valid_indices % width).astype(np.float64) + 0.5
        my = (valid_indices // width).astype(np.float64) + 0.5
        local_x = mx * resolution
        local_y = my * resolution
        yaw = grid_origin_yaw(grid_info)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        origin = grid_info.origin.position
        world_x = float(origin.x) + cos_yaw * local_x - sin_yaw * local_y
        world_y = float(origin.y) + sin_yaw * local_x + cos_yaw * local_y

        # Preserve the legacy behaviour for malformed/truncated confidence
        # arrays: cells without a corresponding confidence entry are omitted
        # from the mean instead of being padded with zeros.
        confidence_values = np.asarray(confidence_data, dtype=np.float64)
        confidence_available = valid_indices < confidence_values.size

        room_statistics = []
        for room_index, raw_room_id in enumerate(room_ids):
            room_id = int(raw_room_id)
            member_mask = room_inverse == room_index
            xs = world_x[member_mask]
            ys = world_y[member_mask]
            cell_count = int(member_mask.sum())
            # Bounds use the midpoint of extrema, not the cell centroid: for
            # an L-shaped room the latter shifts the box off its own cells.
            center = [
                float((xs.min() + xs.max()) * 0.5),
                float((ys.min() + ys.max()) * 0.5),
                0.5 * self.room_box_height,
            ]
            cell_extent = resolution * (abs(cos_yaw) + abs(sin_yaw))
            size = [
                max(resolution, float(xs.max() - xs.min()) + cell_extent),
                max(resolution, float(ys.max() - ys.min()) + cell_extent),
                self.room_box_height,
            ]
            room_confidence_mask = member_mask & confidence_available
            if np.any(room_confidence_mask):
                confidence = float(
                    confidence_values[valid_indices[room_confidence_mask]].mean()
                )
            else:
                confidence = 0.0
            statistic = {
                "room_id": room_id,
                "center": center,
                "size": size,
                "cell_count": cell_count,
                "confidence": confidence,
            }
            room_statistics.append(statistic)
            self._apply_room_grid_statistic(
                statistic,
                geometry_stability_frames,
            )
        self._cache_room_grid_statistics(
            grid_info,
            scene_data,
            confidence_data,
            room_statistics,
        )

    def _refresh_room_nodes_from_cached_stats(
        self, geometry_stability_frames=None, *, count_observation=True
    ):
        cache = self._room_grid_stats_cache
        if cache is None:
            return self._refresh_room_nodes_from_grid(geometry_stability_frames)
        if geometry_stability_frames is None:
            geometry_stability_frames = self.room_geometry_stability_frames
        for statistic in cache.get("statistics") or []:
            self._apply_room_grid_statistic(
                statistic,
                geometry_stability_frames,
                count_observation=count_observation,
            )

    def _apply_room_grid_statistic(
        self, statistic, geometry_stability_frames, *, count_observation=True
    ):
        room_id = int(statistic["room_id"])
        center = list(statistic["center"])
        size = list(statistic["size"])
        # The current grid pass already computed this room's geometry.  Reuse
        # it while creating a previously unseen node; otherwise
        # ``_ensure_room_node`` falls back to two additional Python scans of
        # the complete grid, only for the geometry to be overwritten below.
        node = self._ensure_room_node(
            room_id,
            geometry_override={
                "center": center,
                "aabb_size": size,
                "cell_count": int(statistic.get("cell_count", 0) or 0),
            },
        )
        # Room grids are the temporal observation stream for room identity.
        # Count accepted grid receipts independently of geometry stability so
        # a room can pass the same two-observation gate as object tracks.
        if count_observation:
            node.attributes["room_observation_count"] = int(
                node.attributes.get("room_observation_count", 0) or 0
            ) + 1
            node.attributes["room_consecutive_observations"] = int(
                node.attributes.get("room_consecutive_observations", 0) or 0
            ) + 1
        stable_geometry = self._accept_room_geometry(
            room_id,
            center,
            size,
            geometry_stability_frames,
        )
        if stable_geometry is not None:
            stable_center, stable_size = stable_geometry
            node.centroid = stable_center
            node.aabb_center = stable_center
            node.aabb_size = stable_size
        confidence = float(statistic["confidence"])
        node.confidence = max(node.confidence, confidence / 100.0)
        node.attributes["cell_count"] = int(statistic["cell_count"])
        node.attributes["active"] = True
        node.attributes["room_lifecycle"] = (
            "persistent"
            if _has_persistent_semantic_evidence(node)
            else "active"
        )
        node.attributes.pop("retired_reason", None)
        node.attributes.pop("retired_graph_revision", None)

    def _retire_absent_room_nodes_from_current_grid(self):
        """Retire rooms absent from the latest accepted room grid.

        ``RoomSegmenter`` normally delays a merge until it is stable, but a
        forced refresh and some topology transitions can still transiently
        publish a room grid with only the surviving room.  Keeping the vanished
        room node active produces stale room candidates and orphaned graph
        relations.  Retiring it is reversible: if the same room label appears
        again on a later grid, ``_apply_room_grid_statistic`` reactivates it.
        Formal merge redirects remain the only permanent aliases.
        """

        if not self.room_grid:
            return
        scene_data = self.room_grid.get("scene_data")
        if scene_data is None or len(scene_data) == 0:
            return
        # ``update_room_grid`` has already reduced the accepted grid to one
        # scalar statistic per room.  Reuse that set on the common unchanged
        # OCC heartbeat instead of iterating every cell again (a 1000² grid
        # otherwise adds roughly 170 ms to the room callback).
        cached_statistics = (self._room_grid_stats_cache or {}).get("statistics")
        if isinstance(cached_statistics, list):
            active_room_ids = {
                self._resolve_room_id(statistic.get("room_id"))
                for statistic in cached_statistics
                if isinstance(statistic, dict) and statistic.get("room_id") is not None
            }
        else:
            active_room_ids = {
                self._resolve_room_id(room_id)
                for room_id in scene_data
                if int(room_id) >= 0
            }
        # An all-unknown grid has no evidence that a previously observed room
        # vanished, so retain the previous lifecycle in that case.
        if not active_room_ids:
            return
        for node in self.nodes.values():
            if node.type != "room" or node.room_id is None:
                continue
            room_id = int(node.room_id)
            if room_id in active_room_ids:
                continue
            if node.attributes.get("is_potential_room", False):
                continue
            # A confirmed merge carries stronger provenance than a temporary
            # absence; do not replace its explicit alias metadata.
            if room_id in self.room_redirects:
                node.attributes["active"] = False
                continue
            # A room that passed the two-grid-frame + M1 gate is a persistent
            # semantic landmark.  Temporary absence from the current OCC
            # window only means it is not visible now; it is not evidence that
            # the room ceased to exist.
            if _has_persistent_semantic_evidence(node):
                node.attributes["active"] = True
                node.attributes["room_lifecycle"] = "persistent_unobserved"
                node.attributes.pop("retired_reason", None)
                node.attributes.pop("retired_graph_revision", None)
                continue
            node.attributes["active"] = False
            node.attributes["room_lifecycle"] = "retired"
            node.attributes["retired_reason"] = "absent_from_latest_room_grid"
            node.attributes["retired_graph_revision"] = int(self.graph_revision) + 1

    def _room_node_is_active(self, room_id):
        if room_id is None:
            return False
        try:
            resolved_room_id = self._resolve_room_id(room_id)
        except (TypeError, ValueError):
            return False
        node = self.nodes.get(f"room_{resolved_room_id}")
        return bool(node is not None and node.attributes.get("active", True))

    def _cache_room_grid_statistics(
        self,
        grid_info,
        scene_data,
        confidence_data,
        statistics,
    ):
        self._room_grid_stats_cache = {
            "geometry_key": self._room_grid_geometry_key(grid_info),
            "scene_data": scene_data,
            "confidence_data": confidence_data,
            "statistics": [dict(statistic) for statistic in statistics],
            "redirect_signature": self._room_redirect_signature(),
        }

    def _room_grid_cache_matches(self, grid_info, scene_data, confidence_data):
        cache = self._room_grid_stats_cache
        previous_grid = self.room_grid
        if cache is None or previous_grid is None:
            return False
        if cache.get("geometry_key") != self._room_grid_geometry_key(grid_info):
            return False
        if cache.get("redirect_signature") != self._room_redirect_signature():
            return False
        return self._room_grid_sequences_equal(
            previous_grid.get("scene_data"),
            scene_data,
        ) and self._room_grid_sequences_equal(
            previous_grid.get("confidence_data"),
            confidence_data,
        )

    @staticmethod
    def _room_grid_sequences_equal(previous, incoming):
        if previous is incoming:
            return True
        if previous is None:
            return incoming is None
        if incoming is None:
            return len(previous) == 0
        if isinstance(previous, np.ndarray) or isinstance(incoming, np.ndarray):
            return bool(np.array_equal(previous, incoming))
        return bool(previous == incoming)

    @staticmethod
    def _owned_room_grid_values(values):
        if values is None:
            return []
        # RoomTopology cache results are immutable tuples.  Retaining that
        # exact object makes the next source-N equality check O(1), while raw
        # list inputs keep the historical store-owned copy semantics.
        if isinstance(values, tuple):
            return values
        return list(values)

    def _room_grid_geometry_key(self, grid_info):
        if grid_info is None:
            return None
        origin = getattr(getattr(grid_info, "origin", None), "position", None)
        return (
            int(getattr(grid_info, "width", 0)),
            int(getattr(grid_info, "height", 0)),
            float(getattr(grid_info, "resolution", 0.0)),
            float(getattr(origin, "x", 0.0)),
            float(getattr(origin, "y", 0.0)),
            float(grid_origin_yaw(grid_info)),
            float(self.room_box_height),
        )

    def _room_redirect_signature(self):
        return tuple(
            sorted(
                (int(secondary), int(primary))
                for secondary, primary in self.room_redirects.items()
            )
        )

    @staticmethod
    def _rooms_with_transferred_cells(previous, info, labels):
        """Only observed reassignment, not unknown-map loss, permits shrinkage."""
        old_info = previous["info"]
        resolution = float(info.resolution)
        yaw = grid_origin_yaw(info)
        if resolution <= 0 or abs(float(old_info.resolution) - resolution) > 1e-8:
            return set()
        if abs(grid_origin_yaw(old_info) - yaw) > 1e-8:
            return set()
        old_origin, new_origin = old_info.origin.position, info.origin.position
        dx, dy = old_origin.x - new_origin.x, old_origin.y - new_origin.y
        offsets = ((math.cos(yaw) * dx + math.sin(yaw) * dy) / resolution,
                   (-math.sin(yaw) * dx + math.cos(yaw) * dy) / resolution)
        if any(abs(value - round(value)) > 1e-4 for value in offsets):
            return set()
        x, y = (int(round(value)) for value in offsets)
        old = np.asarray(previous["scene_data"]).reshape(-1, int(old_info.width))
        new = np.asarray(labels).reshape(-1, int(info.width))
        left, bottom = max(0, x), max(0, y)
        right, top = min(new.shape[1], x + old.shape[1]), min(new.shape[0], y + old.shape[0])
        if right <= left or top <= bottom:
            return set()
        old = old[bottom-y:top-y, left-x:right-x]
        new = new[bottom:top, left:right]
        transferred = (old >= 0) & (new >= 0) & (old != new)
        return set(np.unique(old[transferred]).tolist())

    def _accept_room_geometry(self, room_id, center, size, stability_frames):
        candidate = self.room_geometry_candidates.get(room_id)
        if candidate is None:
            candidate = {
                # ``center``/``size`` are a pending relock proposal.  The
                # public room box is intentionally not driven by every room
                # grid publication: exploration grows a room one fringe cell
                # at a time and that made the box visibly walk every frame.
                "center": list(center),
                "size": list(size),
                "count": 0,
                "accepted_center": list(center),
                "accepted_size": list(size),
            }
            self.room_geometry_candidates[room_id] = candidate
            return list(center), list(size)

        # A confirmed split can transfer cells to another room.  Preserve
        # temporal filtering, but allow a stable smaller footprint to replace
        # the historical envelope instead of containing the new room forever.
        shrinks = any(float(size[i]) < float(candidate["accepted_size"][i]) - 1e-6
                      for i in (0, 1))
        if shrinks and room_id in getattr(self, "_room_split_shrink_allowed", set()):
            previous = candidate.get("shrink_proposal")
            if previous and self._room_geometry_relock_close(
                previous["center"], previous["size"], center, size
            ):
                count = previous["count"] + 1
            else:
                count = 1
            candidate["shrink_proposal"] = {
                "center": list(center), "size": list(size), "count": count}
            if count >= max(1, int(stability_frames)):
                candidate.update(center=list(center), size=list(size), count=0,
                                 accepted_center=list(center), accepted_size=list(size))
                candidate.pop("shrink_proposal", None)
                self._room_split_shrink_allowed.discard(room_id)
            return list(candidate["accepted_center"]), list(candidate["accepted_size"])
        candidate.pop("shrink_proposal", None)

        proposed_center, proposed_size = self._expanded_room_geometry(
            candidate["accepted_center"],
            candidate["accepted_size"],
            center,
            size,
        )
        if self._room_geometry_close(
            candidate["accepted_center"],
            candidate["accepted_size"],
            proposed_center,
            proposed_size,
        ) and all(
            abs(float(proposed_size[index]) - float(candidate["accepted_size"][index]))
            <= 1e-6
            for index in (0, 1)
        ):
            # A shrink or an observation already contained by the accepted box
            # is not a new proposal.  Clear any transient expansion streak.
            candidate["center"] = list(candidate["accepted_center"])
            candidate["size"] = list(candidate["accepted_size"])
            candidate["count"] = 0
            self.room_geometry_candidates[room_id] = candidate
            return list(candidate["accepted_center"]), list(candidate["accepted_size"])

        # Exploration normally grows a room by a different fringe on every
        # frame.  Requiring those evolving boxes to be mutually close resets
        # the confirmation streak forever, so the public box can remain stale
        # while the segmentation visibly expands.  Accumulate consecutive
        # expansion observations into one monotonic pending union instead.
        # This still batches public updates and never admits a shrink/jump.
        pending_center, pending_size = self._expanded_room_geometry(
            candidate["center"],
            candidate["size"],
            proposed_center,
            proposed_size,
        )
        candidate["count"] += 1
        candidate["center"] = list(pending_center)
        candidate["size"] = list(pending_size)
        if candidate["count"] >= max(1, int(stability_frames)):
            candidate["accepted_center"] = list(pending_center)
            candidate["accepted_size"] = list(pending_size)
            # A committed checkpoint starts a new streak.  Without this reset,
            # count remains above the threshold and every later frame mutates
            # the public box again.
            candidate["count"] = 0
        self.room_geometry_candidates[room_id] = candidate
        return list(candidate["accepted_center"]), list(candidate["accepted_size"])

    @staticmethod
    def _room_geometry_relock_close(old_center, old_size, center, size):
        """Whether two pending room boxes represent the same stable proposal."""

        center_delta = math.hypot(
            float(old_center[0]) - float(center[0]),
            float(old_center[1]) - float(center[1]),
        )
        size_delta = max(
            abs(float(old_size[0]) - float(size[0])),
            abs(float(old_size[1]) - float(size[1])),
        )
        # Half-cell scale jitter must settle before it changes graph geometry.
        # These limits are deliberately much tighter than identity matching in
        # ``_room_geometry_close``.
        return center_delta <= 0.05 and size_delta <= 0.10

    @staticmethod
    def _room_geometry_close(old_center, old_size, center, size):
        center_delta = math.hypot(
            float(old_center[0]) - float(center[0]),
            float(old_center[1]) - float(center[1]),
        )
        size_delta = max(
            abs(float(old_size[0]) - float(size[0])),
            abs(float(old_size[1]) - float(size[1])),
        )
        return center_delta <= 0.40 and size_delta <= 0.60

    def _resolve_room_id(self, room_id):
        room_id = int(room_id)
        seen = set()
        while room_id in self.room_redirects and room_id not in seen:
            seen.add(room_id)
            room_id = int(self.room_redirects[room_id])
        return room_id

    def _apply_room_merges(self, merges):
        for secondary, primary in merges.items():
            secondary = self._resolve_room_id(secondary)
            primary = self._resolve_room_id(primary)
            if secondary == primary:
                continue
            # Occupancy connected-components cannot distinguish a doorway from
            # a single room after the leaf disappears.  If the graph already
            # knows a portal between these room IDs, retain both stable room
            # identities; otherwise one free-space frame permanently collapses
            # the topology and later graph revisions lose the portal context.
            if self._room_pair_has_known_portal(secondary, primary):
                continue
            self._merge_room_geometry_memory(secondary, primary)
            self.room_redirects[secondary] = primary
            old_node = self.nodes.get(f"room_{secondary}")
            if old_node is not None:
                old_node.attributes["active"] = False
                old_node.attributes["merged_into"] = f"room_{primary}"
            for node in self.nodes.values():
                if node.type != "room" and node.room_id == secondary:
                    node.room_id = primary
                if node.type == "portal":
                    room_ids = node.attributes.get("connected_room_ids") or []
                    node.attributes["connected_room_ids"] = sorted(
                        {self._resolve_room_id(room_id) for room_id in room_ids}
                    )

    def _room_pair_has_known_portal(self, secondary, primary):
        target = {int(secondary), int(primary)}
        if len(target) != 2:
            return False
        for node in self.nodes.values():
            if node.type != "portal":
                continue
            attrs = node.attributes or {}
            observed = attrs.get("observed_connected_room_ids") or []
            connected = attrs.get("connected_room_ids") or []
            for values in (observed, connected):
                resolved = {
                    self._resolve_room_id(value)
                    for value in values
                    if value is not None
                }
                if target.issubset(resolved):
                    return True
        return False

    def _merge_room_geometry_memory(self, secondary, primary):
        """Move both public room extents behind the surviving stable room ID."""

        def accepted_geometry(room_id):
            candidate = self.room_geometry_candidates.get(room_id)
            if candidate is not None:
                return (
                    list(candidate["accepted_center"]),
                    list(candidate["accepted_size"]),
                )
            node = self.nodes.get(f"room_{room_id}")
            if node is not None:
                return list(node.aabb_center), list(node.aabb_size)
            return None

        primary_geometry = accepted_geometry(primary)
        secondary_geometry = accepted_geometry(secondary)
        if primary_geometry is None and secondary_geometry is None:
            return
        if primary_geometry is None:
            merged_center, merged_size = secondary_geometry
        elif secondary_geometry is None:
            merged_center, merged_size = primary_geometry
        else:
            merged_center, merged_size = self._expanded_room_geometry(
                primary_geometry[0],
                primary_geometry[1],
                secondary_geometry[0],
                secondary_geometry[1],
            )
        self.room_geometry_candidates[primary] = {
            "center": list(merged_center),
            "size": list(merged_size),
            "count": 0,
            "accepted_center": list(merged_center),
            "accepted_size": list(merged_size),
        }
        self.room_geometry_candidates.pop(secondary, None)
        primary_node = self.nodes.get(f"room_{primary}")
        if primary_node is not None:
            primary_node.centroid = list(merged_center)
            primary_node.aabb_center = list(merged_center)
            primary_node.aabb_size = list(merged_size)

    def _refresh_missing_room_nodes_from_observations(self):
        # Room IDs present in the latest occupancy-derived statistics already
        # have their temporal observation count advanced by
        # ``update_room_grid``.  Detector frames may reuse those statistics at
        # 10 Hz; counting them here would let YOLO cadence satisfy the
        # two-room-grid-frame admission gate.  Only genuinely provisional
        # rooms (with no occupancy geometry yet) may use object observations
        # as their fallback evidence.
        grid_room_ids = {
            int(statistic.get("room_id"))
            for statistic in (self._room_grid_stats_cache or {}).get("statistics", [])
            if isinstance(statistic, dict) and statistic.get("room_id") is not None
        }
        room_to_nodes = defaultdict(list)
        for node in self.nodes.values():
            if node.type == "room" or node.room_id is None:
                continue
            if int(node.room_id) in self.room_geometries:
                continue
            if self._resolve_room_id(node.room_id) in grid_room_ids:
                continue
            room_to_nodes[int(node.room_id)].append(node)
        for room_id, child_nodes in room_to_nodes.items():
            if not child_nodes:
                continue
            room_node = self._ensure_room_node(room_id)
            mins = []
            maxs = []
            for child in child_nodes:
                center = [float(v) for v in child.aabb_center]
                size = [float(v) for v in child.aabb_size]
                mins.append([center[i] - size[i] * 0.5 for i in range(3)])
                maxs.append([center[i] + size[i] * 0.5 for i in range(3)])
            min_corner = [min(point[i] for point in mins) for i in range(3)]
            max_corner = [max(point[i] for point in maxs) for i in range(3)]
            center = [(min_corner[i] + max_corner[i]) * 0.5 for i in range(3)]
            center[2] = 0.5 * self.room_box_height
            size = [max(max_corner[i] - min_corner[i], 0.1) for i in range(3)]
            size[2] = self.room_box_height
            room_node.attributes["room_observation_count"] = int(
                room_node.attributes.get("room_observation_count", 0) or 0
            ) + 1
            room_node.attributes["room_consecutive_observations"] = int(
                room_node.attributes.get("room_consecutive_observations", 0) or 0
            ) + 1
            stable_center, stable_size = self._accept_room_geometry(
                room_id,
                center,
                size,
                self.room_geometry_stability_frames,
            )
            room_node.centroid = stable_center
            room_node.aabb_center = stable_center
            room_node.aabb_size = stable_size
            room_node.attributes["estimated_from_observations"] = True
            room_node.attributes["cell_count"] = len(child_nodes)

    def _ensure_room_node(self, room_id, geometry_override=None):
        room_id = int(room_id)
        node_id = f"room_{room_id}"
        node = self.nodes.get(node_id)
        if node is None:
            label = room_node_label(self.room_id_to_name, room_id)
            geometry = dict(self.room_geometries.get(room_id, {}) or {})
            if geometry_override:
                # Existing explicit room geometry remains authoritative.  The
                # override only fills fields needed for first construction.
                for key, value in dict(geometry_override).items():
                    if value is not None:
                        geometry.setdefault(key, value)
            center = self._grounded_center(geometry.get("center") or self._default_room_center(room_id))
            aabb_center = self._grounded_center(geometry.get("aabb_center") or center)
            size = self._room_box_size(geometry.get("aabb_size") or self._default_room_size(room_id))
            node = SceneGraphNode(
                id=node_id,
                type="room",
                label=label,
                name=str(geometry.get("name") or f"{label}_{room_id}"),
                centroid=center,
                aabb_center=aabb_center,
                aabb_size=size,
                parent_id=self._ensure_scene_node().id,
                room_id=room_id,
            )
            if "cell_count" in geometry:
                node.attributes["cell_count"] = int(geometry["cell_count"])
            self.nodes[node_id] = node
        return node

    def ensure_provisional_room_for_portal(self, portal_node):
        """Create a planning-only room behind a stable physical portal."""
        if not self.force_provisional_portal_rooms or portal_node is None:
            return None
        if getattr(portal_node, "type", "") != "portal":
            return None
        # Do not manufacture a room from a raw YOLO door hypothesis.  The
        # portal must first pass the same two-frame + M1 persistence gate as
        # every other semantic landmark; otherwise every transient door box
        # creates an overlapping synthetic room.
        if not _has_persistent_semantic_evidence(portal_node):
            return None
        attrs = portal_node.attributes
        room_id = self._ensure_open_portal_child_room(
            portal_node, portal_node.interaction.get("operation_history") or [],
            geometry_source="portal_prior",
        )
        if room_id is None:
            return None
        room = self.nodes[f"room_{room_id}"]
        room.attributes.update({
            "active": True,
            "is_potential_room": True,
            "room_lifecycle": "provisional",
            "room_assignment_source": "portal_prior",
            "confirmed": False,
            "observed_free_space": False,
            "parent_portal_id": portal_node.id,
        })
        attrs["potential_room_ids"] = [room_id]
        attrs["potential_room_source"] = "forced_portal_room"
        attrs["potential_room_confirmed"] = False
        return room

    def prune_unqualified_provisional_rooms(self):
        """Remove synthetic door-side rooms whose portal lacks M1 evidence."""

        invalid_room_ids = []
        for room_id, room in self.nodes.items():
            if room.type != "room" or not room.attributes.get("is_potential_room"):
                continue
            portal_id = str(
                room.attributes.get("source_portal_id")
                or room.attributes.get("parent_portal_id")
                or ""
            )
            portal = self.nodes.get(portal_id)
            if portal is None or not _has_persistent_semantic_evidence(portal):
                invalid_room_ids.append(room_id)
        if not invalid_room_ids:
            return False
        invalid = set(invalid_room_ids)
        invalid_numeric_ids = {int(self.nodes[node_id].room_id) for node_id in invalid}
        for room_id in invalid_room_ids:
            self.nodes.pop(room_id, None)
        self.edges = {
            edge_id: edge
            for edge_id, edge in self.edges.items()
            if edge.src_id not in invalid and edge.dst_id not in invalid
        }
        for portal in self.nodes.values():
            potential_ids = [
                int(value)
                for value in (portal.attributes.get("potential_room_ids") or [])
                if str(value).lstrip("-").isdigit() and int(value) not in invalid_numeric_ids
            ]
            if potential_ids:
                portal.attributes["potential_room_ids"] = potential_ids
            else:
                portal.attributes.pop("potential_room_ids", None)
                portal.attributes.pop("potential_room_source", None)
                portal.attributes.pop("potential_room_confirmed", None)
                portal.attributes.pop("portal_child_room_id", None)
                portal.attributes.pop("portal_child_source_room_id", None)
        self._rebuild_relations()
        self._bump_revision()
        return True

    def _default_room_center(self, room_id):
        if not self.room_grid:
            return [float(room_id), 0.0, 0.0]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [float(room_id), 0.0, 0.0]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            wx, wy = grid_to_world(mx, my, grid_info)
            xs.append(wx)
            ys.append(wy)
        if xs and ys:
            return [sum(xs) / len(xs), sum(ys) / len(ys), 0.5 * self.room_box_height]
        return [float(room_id), 0.0, 0.0]

    def _default_room_size(self, room_id):
        if not self.room_grid:
            return [0.5, 0.5, 0.1]
        grid_info = self.room_grid.get("info")
        scene_data = list(self.room_grid.get("scene_data") or [])
        if grid_info is None or not scene_data:
            return [0.5, 0.5, 0.1]
        width = int(grid_info.width)
        xs = []
        ys = []
        for idx, scene_id in enumerate(scene_data):
            if int(scene_id) != int(room_id):
                continue
            mx = idx % width
            my = idx // width
            wx, wy = grid_to_world(mx, my, grid_info)
            xs.append(wx)
            ys.append(wy)
        if len(xs) > 1 and len(ys) > 1:
            return [max(max(xs) - min(xs), float(grid_info.resolution)), max(max(ys) - min(ys), float(grid_info.resolution)), self.room_box_height]
        resolution = float(grid_info.resolution) if grid_info is not None else 0.5
        return [resolution, resolution, self.room_box_height]

    def _ground_non_room_centroid(self, center, size):
        grounded = [float(center[0]), float(center[1]), float(center[2])]
        half_height = max(float(size[2]) * 0.5, 0.01)
        grounded[2] = max(grounded[2], half_height)
        return grounded

    @staticmethod
    def _valid_aabb_size(size) -> bool:
        try:
            values = [float(value) for value in list(size or [])[:3]]
        except (TypeError, ValueError):
            return False
        return len(values) == 3 and all(
            math.isfinite(value) and value > 1e-6 for value in values
        )

    def _grounded_center(self, center):
        grounded = list(center)
        if len(grounded) < 3:
            grounded.extend([0.0] * (3 - len(grounded)))
        grounded = [float(v) for v in grounded[:3]]
        grounded[2] = 0.5 * self.room_box_height
        return grounded

    def _room_box_size(self, size):
        size = list(size or [])
        if len(size) < 3:
            size.extend([0.0] * (3 - len(size)))
        room_size = [max(float(size[0]), 0.1), max(float(size[1]), 0.1), self.room_box_height]
        return room_size

    @staticmethod
    def _distinct_room_ids(values):
        result = []
        for value in values or []:
            try:
                room_id = int(value)
            except (TypeError, ValueError):
                continue
            if room_id < 0 or room_id in result:
                continue
            result.append(room_id)
        return result

    def _portal_observed_room_ids(self, node):
        """Return only room IDs supported by observations/segmentation.

        ``connected_room_ids`` is intentionally not read here because it is
        also the public topology field and can contain a synthetic child room
        after a successful open.  Keeping this source separate prevents a
        potential child from being mistaken for observed free space on later
        graph refreshes.
        """

        direct = self._distinct_room_ids(
            node.attributes.get("observed_connected_room_ids") or []
        )
        if self.source_mode == "gt_replay" and direct:
            return direct[:2]
        inferred = self._distinct_room_ids(self._infer_portal_room_ids(node))
        if inferred:
            return inferred[:2]
        if node.room_id is not None:
            return [self._resolve_room_id(node.room_id)]
        return []

    def _allocate_portal_child_room_id(self):
        while True:
            room_id = int(self.next_portal_child_room_id)
            self.next_portal_child_room_id += 1
            if f"room_{room_id}" not in self.nodes:
                return room_id

    def _portal_child_direction(self, node, source_room_id, history):
        """Estimate the side reached by crossing a just-opened portal.

        A successful interaction approach is the most reliable local signal:
        from approach pose to door center points through the doorway.  If an
        older result lacks that pose, use the known source-room centroid; the
        final AABB-normal fallback only gives the potential room a stable
        visualization/association anchor and is explicitly marked as such.
        """

        center = list(
            node.attributes.get("interaction_reference_aabb_center")
            or node.aabb_center
            or node.centroid
            or []
        )
        if len(center) < 2:
            return (1.0, 0.0, "fallback_axis")
        center_x, center_y = float(center[0]), float(center[1])
        for event in reversed(list(history or [])):
            if not bool(event.get("success")):
                continue
            approach = list(event.get("approach_goal_xyyaw") or [])
            if len(approach) < 2:
                continue
            dx = center_x - float(approach[0])
            dy = center_y - float(approach[1])
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                return (dx / norm, dy / norm, "interaction_approach")
        source_room = self.nodes.get(f"room_{int(source_room_id)}")
        if source_room is not None:
            source_center = list(
                source_room.aabb_center or source_room.centroid or []
            )
            if len(source_center) >= 2:
                dx = center_x - float(source_center[0])
                dy = center_y - float(source_center[1])
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    return (dx / norm, dy / norm, "source_room_centroid")
        size = list(node.aabb_size or [])
        size_x = abs(float(size[0])) if len(size) >= 1 else 0.0
        size_y = abs(float(size[1])) if len(size) >= 2 else 0.0
        # The long AABB axis approximates the door span, so cross it to get
        # a deterministic normal when neither side was otherwise observed.
        if size_x >= size_y:
            return (0.0, 1.0, "door_aabb_normal")
        return (1.0, 0.0, "door_aabb_normal")

    def _ensure_open_portal_child_room(self, node, history, *, geometry_source="portal_open_potential_child"):
        """Create one unobserved child room after a portal changes state.

        This is deliberately a graph-only room.  It provides a stable target
        for the post-door transition and nearby frontier association, while
        leaving every unknown occupancy-grid cell at ``room_unknown_id``.
        """

        # An open-looking crop or an executor event is not enough to invent a
        # room. Require the portal's two-frame + M1 semantic confirmation so
        # transient door detections cannot leave overlapping child regions.
        if not _has_persistent_semantic_evidence(node):
            return None

        observed_room_ids = self._portal_observed_room_ids(node)
        if len(observed_room_ids) >= 2:
            return None
        source_room_id = (
            observed_room_ids[0]
            if observed_room_ids
            else self._resolve_room_id(node.room_id)
            if node.room_id is not None
            else None
        )
        if source_room_id is None:
            return None
        attributes = node.attributes
        existing = attributes.get("portal_child_room_id")
        if existing is None:
            # Adopt the earlier forced-room representation instead of
            # allocating a second room when its door is subsequently opened.
            for candidate_id in attributes.get("potential_room_ids") or []:
                candidate = self.nodes.get(f"room_{candidate_id}")
                if candidate is not None and candidate.attributes.get("is_potential_room"):
                    owner = candidate.attributes.get("source_portal_id") or candidate.attributes.get("parent_portal_id")
                    if owner == node.id:
                        existing = candidate.room_id
                        break
        try:
            child_room_id = int(existing)
        except (TypeError, ValueError):
            child_room_id = None
        if child_room_id is not None:
            child = self.nodes.get(f"room_{child_room_id}")
            if child is not None and child.attributes.get("resolved_to_room_id") is not None:
                return None
            if child is not None and (
                child.attributes.get("active", True)
                or child.attributes.get("is_potential_room", False)
                or child.attributes.get("room_lifecycle") == "provisional"
            ):
                # A provisional child is graph-only until OCC observes free
                # space on the far side. Older graph revisions could have
                # retired it during a transient room-grid refresh; revive the
                # same ID instead of allocating a second synthetic room.
                child.attributes.update(
                    {
                        "active": True,
                        "is_potential_room": True,
                        "room_lifecycle": "provisional",
                        "confirmed": False,
                        "observed_free_space": False,
                        "source_portal_id": node.id,
                    }
                )
                child.attributes.pop("retired_reason", None)
                child.attributes.pop("retired_graph_revision", None)
                attributes["portal_child_source_room_id"] = int(source_room_id)
                attributes["portal_child_room_id"] = child_room_id
                attributes["potential_room_ids"] = [child_room_id]
                existing_geometry_source = child.attributes.get("room_geometry_source")
                if existing_geometry_source in {geometry_source, "portal_open_potential_child"}:
                    return child_room_id
                # Legacy forced rooms used room ID as X. Repair their
                # geometry in place; an actual opening can also refine the
                # prior side using its approach pose without changing IDs.
            else:
                child_room_id = None
        if child_room_id is None:
            child_room_id = self._allocate_portal_child_room_id()
        unit_x, unit_y, direction_source = self._portal_child_direction(
            node, source_room_id, history
        )
        center = list(
            attributes.get("interaction_reference_aabb_center")
            or node.aabb_center
            or node.centroid
            or [0.0, 0.0, 0.0]
        )
        center.extend([0.0] * max(0, 3 - len(center)))
        portal_size = list(
            attributes.get("interaction_reference_aabb_size") or node.aabb_size or []
        )
        portal_size.extend([0.0] * max(0, 3 - len(portal_size)))
        portal_span_m = max(abs(float(portal_size[0])), abs(float(portal_size[1])))
        width_m = max(self.portal_child_room_min_width_m, portal_span_m + 0.4)
        depth_m = self.portal_child_room_depth_m
        # Axis-aligned projection of a small oriented doorway-side rectangle.
        size_x = max(0.4, abs(unit_x) * depth_m + abs(unit_y) * width_m)
        size_y = max(0.4, abs(unit_y) * depth_m + abs(unit_x) * width_m)
        child_center = [
            float(center[0]) + unit_x * self.portal_child_room_offset_m,
            float(center[1]) + unit_y * self.portal_child_room_offset_m,
            0.5 * self.room_box_height,
        ]
        self.room_id_to_name.setdefault(child_room_id, "unobserved_portal_room")
        child = self._ensure_room_node(child_room_id, geometry_override={
            "center": child_center, "aabb_center": child_center,
            "aabb_size": [size_x, size_y, self.room_box_height], "cell_count": 0,
        })
        child.name = f"unobserved_room_beyond_{node.id}"
        child.centroid = list(child_center)
        child.aabb_center = list(child_center)
        child.aabb_size = [size_x, size_y, self.room_box_height]
        child.confidence = max(float(child.confidence), 0.25)
        child.attributes.update(
            {
                "active": True,
                "is_potential_room": True,
                "observed_free_space": False,
                "cell_count": 0,
                "source_portal_id": node.id,
                "source_room_id": int(source_room_id),
                "portal_child_direction_xy": [round(unit_x, 4), round(unit_y, 4)],
                "portal_child_direction_source": direction_source,
                "room_geometry_source": geometry_source,
            }
        )
        attributes.update(
            {
                "portal_child_room_id": child_room_id,
                "portal_child_source_room_id": int(source_room_id),
                "potential_room_ids": [child_room_id],
            }
        )
        return child_room_id

    def _resolve_portal_child_room(self, node, observed_room_ids):
        """Retire a synthetic child once segmentation observes its far side."""

        attributes = node.attributes
        try:
            child_room_id = int(attributes.get("portal_child_room_id"))
        except (TypeError, ValueError):
            return None
        try:
            source_room_id = int(attributes.get("portal_child_source_room_id"))
        except (TypeError, ValueError):
            source_room_id = None
        observed_room_ids = self._distinct_room_ids(observed_room_ids)
        actual_room_id = next(
            (
                room_id
                for room_id in observed_room_ids
                if room_id != source_room_id and room_id != child_room_id
            ),
            None,
        )
        if actual_room_id is None:
            return None
        child = self.nodes.get(f"room_{child_room_id}")
        if child is not None:
            child.attributes.update(
                {
                    "active": False,
                    "resolved_to_room_id": int(actual_room_id),
                    "resolved_from_observed_free_space": True,
                }
            )
        self.room_redirects[child_room_id] = int(actual_room_id)
        attributes.update(
            {
                "resolved_portal_child_room_id": child_room_id,
                "portal_child_room_id": None,
                "potential_room_ids": [],
            }
        )
        return int(actual_room_id)

    def _rebuild_relations(self, now=None):
        now = float(now if now is not None else time.time())
        self.edges = {}
        scene_node = self._ensure_scene_node()
        scene_node.attributes["source_mode"] = self.source_mode
        scene_node.attributes["episode_id"] = self.episode_id
        # A delayed M1 answer can arrive after a successful refrigerator open
        # and temporarily rewrite the mutable graph type to ``object``.  The
        # source refrigerator identity plus the sealed success history are
        # stronger evidence than that one-frame class result.  Restore the
        # container type before building the relation pools so the newly
        # exposed contents are not silently left parentless.
        for node in self.nodes.values():
            _restore_confirmed_refrigerator_container_type(node)
        rooms = {
            node.id: node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        }
        for room_node in (
            node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        ):
            room_node.parent_id = scene_node.id
            self._upsert_edge(scene_node.id, "has_room", room_node.id, now=now)
        non_rooms = [node for node in self.nodes.values() if node.type not in {"scene", "room"}]
        previous_parent_ids = {node.id: node.parent_id for node in non_rooms}

        for node in non_rooms:
            if node.type == "portal":
                node.parent_id = scene_node.id
            room_id = node.room_id
            # An explicit first observation may introduce a room before OCC.
            # Missing is different from an existing, deliberately retired room.
            if room_id is not None and f"room_{self._resolve_room_id(room_id)}" not in self.nodes:
                self._ensure_room_node(self._resolve_room_id(room_id))
            # A node can retain a room label from a previous segmentation
            # revision even after that room has been retired.  Re-sample the
            # current room grid before building relations so candidates do not
            # keep targeting an absent ``room_N``.  This is deliberately
            # reversible; a confirmed merge still owns the persistent alias.
            if room_id is None or not self._room_node_is_active(room_id):
                inferred_room_id = self._infer_room_id_from_node(node)
                if inferred_room_id is not None:
                    room_id = self._resolve_room_id(inferred_room_id)
                    node.room_id = room_id
                    node.attributes["room_assignment_source"] = "latest_room_grid"
                    node.attributes.pop("room_assignment_stale", None)
                elif room_id is not None:
                    node.attributes["room_assignment_stale"] = True
            if room_id is not None and self._room_node_is_active(room_id):
                room_id = self._resolve_room_id(room_id)
                node.room_id = room_id
                room_node = self._ensure_room_node(room_id)
                if node.type != "portal":
                    node.parent_id = room_node.id
                self._upsert_edge(node.id, "in_room", room_node.id, now=now)
                self._upsert_edge(room_node.id, "has_child", node.id, now=now)
            elif node.type != "portal":
                # Avoid an in-room edge to a retired node when the current map
                # has no grounded alternative for this object yet.
                node.parent_id = scene_node.id

        for node in non_rooms:
            if node.type == "portal":
                observed_room_ids = [
                    self._resolve_room_id(room_id)
                    for room_id in self._portal_observed_room_ids(node)
                ]
                observed_room_ids = self._distinct_room_ids(observed_room_ids)
                try:
                    child_source_room_id = int(
                        node.attributes.get("portal_child_source_room_id")
                    )
                except (TypeError, ValueError):
                    child_source_room_id = None
                if child_source_room_id in observed_room_ids:
                    observed_room_ids = [child_source_room_id] + [
                        room_id
                        for room_id in observed_room_ids
                        if room_id != child_source_room_id
                    ]
                # If a new free-space component has appeared on the far side,
                # replace the graph-only child with the occupancy-derived
                # room.  Until then, retain the child as topology only.
                self._resolve_portal_child_room(node, observed_room_ids)
                potential_room_ids = self._distinct_room_ids(
                    node.attributes.get("potential_room_ids") or []
                )
                connected_room_ids = list(observed_room_ids)
                for room_id in potential_room_ids:
                    potential_node = self.nodes.get(f"room_{room_id}")
                    if potential_node is None or not potential_node.attributes.get(
                        "active", True
                    ):
                        continue
                    if room_id not in connected_room_ids:
                        connected_room_ids.append(room_id)
                node.attributes["connected_room_ids"] = connected_room_ids
                node.attributes["observed_connected_room_ids"] = list(
                    observed_room_ids
                )
                node.attributes["potential_room_ids"] = list(potential_room_ids)
                if potential_room_ids:
                    node.attributes["connectivity_source"] = (
                        "portal_open_potential_child"
                    )
                    node.attributes["connectivity_status"] = "potential"
                else:
                    node.attributes["connectivity_source"] = (
                        "observation"
                        if self.source_mode == "gt_replay" and observed_room_ids
                        else "room_segment_ring"
                    )
                    node.attributes["connectivity_status"] = (
                        "connected"
                        if len(connected_room_ids) >= 2
                        else "partial"
                        if len(connected_room_ids) == 1
                        else "unknown"
                    )
                traversable = node.interaction.get("traversable")
                edge_attributes = {
                    "portal_node_id": node.id,
                    "state": node.interaction.get("state", "unknown"),
                    "traversable": traversable,
                    "requires_interaction": bool(node.interaction.get("requires_interaction")),
                    "interaction_mode": node.interaction.get("interaction_mode", "none"),
                    "interaction_cost": float(node.interaction.get("interaction_cost", 1.0)),
                    "expected_effect": "unlock_connectivity",
                    "connectivity_status": node.attributes["connectivity_status"],
                    "observed_connected_room_ids": list(observed_room_ids),
                    "potential_room_ids": list(potential_room_ids),
                    "potential_connectivity": bool(potential_room_ids),
                }
                for room_id in sorted(set(int(room) for room in connected_room_ids if room is not None)):
                    room_node = self._ensure_room_node(room_id)
                    self._upsert_edge(node.id, "connects", room_node.id, attributes=edge_attributes, now=now)
                    self._upsert_edge(room_node.id, "adjacent_via", node.id, attributes=edge_attributes, now=now)

        support_nodes = [node for node in non_rooms if node.type == "support"]
        container_nodes = [node for node in non_rooms if node.type == "container"]
        object_nodes = [node for node in non_rooms if node.type == "object"]
        # Parent inference accepts a support/container from the same room or
        # from an unknown-room candidate. Bucket known rooms while retaining
        # the unknown bucket as the conservative historical fallback. The
        # stable merge below preserves non_rooms insertion order for ties.
        support_by_room = defaultdict(list)
        container_by_room = defaultdict(list)
        for candidate in support_nodes:
            support_by_room[candidate.room_id].append(candidate)
        for candidate in container_nodes:
            container_by_room[candidate.room_id].append(candidate)
        relation_order = {node.id: index for index, node in enumerate(non_rooms)}

        def room_relation_candidates(by_room, all_candidates, room_id):
            if room_id is None:
                return all_candidates
            room_candidates = by_room.get(room_id, [])
            unknown_candidates = by_room.get(None, [])
            if not unknown_candidates:
                return room_candidates
            if not room_candidates:
                return unknown_candidates
            merged = []
            room_index = 0
            unknown_index = 0
            while room_index < len(room_candidates) or unknown_index < len(unknown_candidates):
                if unknown_index >= len(unknown_candidates):
                    merged.append(room_candidates[room_index])
                    room_index += 1
                elif room_index >= len(room_candidates):
                    merged.append(unknown_candidates[unknown_index])
                    unknown_index += 1
                elif relation_order[room_candidates[room_index].id] < relation_order[
                    unknown_candidates[unknown_index].id
                ]:
                    merged.append(room_candidates[room_index])
                    room_index += 1
                else:
                    merged.append(unknown_candidates[unknown_index])
                    unknown_index += 1
            return merged

        # A delayed M1 response can temporarily relabel a bottle/food track as
        # ``container``.  It is still a possible refrigerator child, whereas
        # portals/support surfaces are never content.  Keep this relaxed pool
        # private to the refrigerator fallback; the strict hierarchy below
        # continues to use the source-observed object type.
        refrigerator_content_nodes = [
            node
            for node in non_rooms
            if node.type not in {"scene", "room", "portal", "support"}
        ]
        id_lookup = {node.id: node for node in non_rooms}
        # A physical refrigerator often reports its contents in the open-door
        # volume rather than in the closed appliance AABB.  The executor has
        # already established the open state at this point, so derive one
        # stable local-depth side from the currently tracked objects.  This is
        # deliberately scoped to refrigerators; drawers, cabinets and closed
        # containers retain the strict AABB rule below.
        open_refrigerator_sides = {}
        for container in container_nodes:
            # These fields describe the active inference only.  Retain the
            # selected side as historical evidence, but do not leave a stale
            # ``open`` mode visible after a close/unknown observation.
            container.attributes.pop("containment_inference_mode", None)
            container.attributes.pop("open_refrigerator_content_side_confidence", None)
            if not _is_open_refrigerator(container):
                continue
            side, confidence = _infer_open_refrigerator_content_side(
                container, refrigerator_content_nodes
            )
            if side is None:
                continue
            open_refrigerator_sides[container.id] = side
            container.attributes.update(
                {
                    "open_refrigerator_content_side": (
                        "positive_depth" if side > 0 else "negative_depth"
                    ),
                    "open_refrigerator_content_side_confidence": float(
                        confidence
                    ),
                    "containment_inference_mode": "open_refrigerator_depth",
                }
            )
        for container in container_nodes:
            container.attributes["inferred_child_ids"] = []
            container.attributes["inferred_child_count"] = 0

        # A confident M1 crop may temporarily change a content track's graph
        # type from ``object`` to ``container``.  Once an opened refrigerator
        # has a selected depth side, retain those tracks in the relation pass
        # when their labels still look like a movable item.  Do not broaden
        # the ordinary (closed-container) hierarchy with this fallback.
        relation_object_nodes = list(object_nodes)
        if open_refrigerator_sides:
            relation_object_ids = {node.id for node in relation_object_nodes}
            for candidate in refrigerator_content_nodes:
                if (
                    candidate.id in relation_object_ids
                    or not _is_open_refrigerator_content_candidate(candidate)
                ):
                    continue
                relation_object_nodes.append(candidate)
                relation_object_ids.add(candidate.id)

        # For the ordinary (closed-container) hierarchy, parent inference is
        # deterministic for a fixed candidate geometry.  Keep one bounded
        # per-context cache so a repeated detector frame does not rescan every
        # container/support for every object.  Open-refrigerator depth
        # inference has additional side evidence and intentionally bypasses
        # this cache.
        parent_context_key = None
        if not open_refrigerator_sides:
            parent_context_key = self._parent_candidate_context_key(
                support_nodes, container_nodes
            )
            if parent_context_key != self._parent_relation_cache_context:
                self._parent_relation_cache.clear()
                self._parent_relation_cache_context = parent_context_key

        next_parent_cache = {}
        for obj in relation_object_nodes:
            previous_parent_id = previous_parent_ids.get(obj.id)
            support_candidates = room_relation_candidates(
                support_by_room, support_nodes, obj.room_id
            )
            container_candidates = room_relation_candidates(
                container_by_room, container_nodes, obj.room_id
            )
            parent = None
            object_cache_key = None
            cached_parent_id = _PARENT_CACHE_MISS
            if parent_context_key is not None:
                object_cache_key = self._parent_object_context_key(
                    obj, previous_parent_id
                )
                cached_parent_id = self._parent_relation_cache.get(
                    (obj.id, object_cache_key), _PARENT_CACHE_MISS
                )
                if cached_parent_id is not _PARENT_CACHE_MISS:
                    parent = id_lookup.get(cached_parent_id)
                    if parent is not None and parent.type not in {
                        "container",
                        "support",
                    }:
                        parent = None
            if object_cache_key is None or cached_parent_id is _PARENT_CACHE_MISS:
                parent = self._find_parent_node(
                    obj,
                    support_candidates,
                    container_candidates,
                    id_lookup,
                    previous_parent_id=previous_parent_id,
                    open_refrigerator_sides=open_refrigerator_sides,
                )
            if object_cache_key is not None:
                next_parent_cache[(obj.id, object_cache_key)] = (
                    parent.id if parent is not None else None
                )
            if parent is None:
                # The first pass already selected an active room or scene.
                # Do not resurrect a retired room from a stale room_id here.
                continue
            obj.parent_id = parent.id
            if parent.type == "support":
                self._upsert_edge(parent.id, "supports", obj.id, now=now)
            elif parent.type == "container":
                edge_attributes = {}
                if (
                    parent.id in open_refrigerator_sides
                    and not _container_contains(obj, parent)
                ):
                    edge_attributes["containment_source"] = (
                        "open_refrigerator_depth_inference"
                    )
                self._upsert_edge(
                    parent.id,
                    "contains",
                    obj.id,
                    attributes=edge_attributes,
                    now=now,
                )
                parent.attributes["inferred_child_ids"].append(obj.id)

        # Only the current context for each live object can be reused on the
        # next rebuild. Historical positions otherwise grow without bound.
        self._parent_relation_cache = next_parent_cache
        for container in container_nodes:
            child_ids = sorted(set(container.attributes["inferred_child_ids"]))
            container.attributes["inferred_child_ids"] = child_ids
            container.attributes["inferred_child_count"] = len(child_ids)

        for room_node in (
            node
            for node in self.nodes.values()
            if node.type == "room" and node.attributes.get("active", True)
        ):
            room_node.parent_id = scene_node.id
            self._upsert_edge(scene_node.id, "has_room", room_node.id, now=now)
        self._refresh_room_attributes()

    @staticmethod
    def _parent_geometry_values(values):
        try:
            return tuple(float(value) for value in list(values or [])[:3])
        except (TypeError, ValueError):
            return ()

    @classmethod
    def _parent_node_context_key(cls, node):
        """Return geometry/semantic fields used by ordinary parent tests."""

        attributes = node.attributes or {}
        return (
            str(node.id),
            str(node.type),
            normalize_label(node.label or node.name),
            node.room_id,
            cls._parent_geometry_values(node.aabb_center or node.centroid),
            cls._parent_geometry_values(node.aabb_size),
            cls._parent_geometry_values(node.centroid),
            bool(node.is_currently_visible),
            str(attributes.get("interaction_state") or ""),
            str(node.interaction.get("state") or ""),
        )

    def _parent_candidate_context_key(self, support_nodes, container_nodes):
        # Preserve insertion order because the historical matcher used that
        # order to break equal-distance/volume ties.
        return tuple(
            self._parent_node_context_key(node)
            for node in (*support_nodes, *container_nodes)
        )

    def _parent_object_context_key(self, obj, previous_parent_id):
        return (
            self._parent_node_context_key(obj),
            # The fallback parent is consulted only for an invisible object;
            # including it for visible tracks would miss the cache on the
            # first frame after a normal parent assignment.
            str(previous_parent_id or "")
            if not bool(obj.is_currently_visible)
            else "",
        )

    def _find_parent_node(
        self,
        obj,
        support_nodes,
        container_nodes,
        id_lookup,
        previous_parent_id=None,
        open_refrigerator_sides=None,
    ):
        # Without a fresh object observation, changing container geometry is
        # not evidence that its contents moved. Preserve the previous parent
        # before considering a newly overlapping neighboring container.
        if not obj.is_currently_visible and previous_parent_id:
            previous_parent = id_lookup.get(str(previous_parent_id))
            if (previous_parent is not None and previous_parent.type == "container"
                and _same_room_or_unknown(obj, previous_parent)):
                return previous_parent
        # An opened refrigerator's measured body box is often the *closed*
        # appliance box.  Its newly visible contents therefore have to win
        # before a stale/nearby closed container or support claims the same
        # object by ordinary AABB proximity.
        open_matches = []
        for container in container_nodes:
            if container.id == obj.id:
                continue
            side = (open_refrigerator_sides or {}).get(container.id)
            if side is None:
                continue
            if _is_open_refrigerator_content(obj, container, side):
                score = _open_refrigerator_content_score(obj, container, side)
                open_matches.append((score, container))
        if open_matches:
            return min(open_matches, key=lambda item: (item[0], volume(item[1].aabb_size)))[1]

        containing = [
            node
            for node in container_nodes
            if node.id != obj.id
            if _is_plausible_container_content(obj, node)
            and self._is_inside_volume(obj, node)
        ]
        if containing:
            return sorted(containing, key=lambda node: volume(node.aabb_size))[0]

        supporting = [node for node in support_nodes if self._is_on_support(obj, node)]
        if supporting:
            return sorted(supporting, key=lambda node: abs(top_surface_z(node) - obj.centroid[2]))[0]
        return None

    def _upsert_edge(self, src_id, relation, dst_id, attributes=None, confidence=1.0, now=None):
        edge_id = f"edge_{relation}_{sanitize_token(src_id)}_{sanitize_token(dst_id)}"
        self.edges[edge_id] = SceneGraphEdge(
            id=edge_id,
            src_id=src_id,
            relation=relation,
            dst_id=dst_id,
            attributes=dict(attributes or {}),
            confidence=float(confidence),
            last_seen=now,
        )

    def _infer_room_id_from_node(self, node):
        if not self.room_grid:
            return None
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        if grid_info is None or not scene_data:
            return None
        candidates = defaultdict(int)
        center = node.aabb_center
        size = node.aabb_size
        half_x = max(float(size[0]) * 0.45, 0.02)
        half_y = max(float(size[1]) * 0.45, 0.02)
        sample_points = [
            (float(center[0]), float(center[1])),
            (float(center[0]) - half_x, float(center[1]) - half_y),
            (float(center[0]) - half_x, float(center[1]) + half_y),
            (float(center[0]) + half_x, float(center[1]) - half_y),
            (float(center[0]) + half_x, float(center[1]) + half_y),
        ]
        for px, py in sample_points:
            coords = world_to_grid(px, py, grid_info)
            if coords is None:
                continue
            idx = grid_index(coords[0], coords[1], grid_info.width)
            if idx < 0 or idx >= len(scene_data):
                continue
            room_id = int(scene_data[idx])
            if room_id >= 0:
                candidates[room_id] += 1
        if candidates:
            return max(sorted(candidates.keys()), key=lambda room_id: candidates[room_id])
        resolution = max(float(getattr(grid_info, "resolution", 0.05)), 0.01)
        margins = []
        margin = max(0.10, resolution * 2.0)
        while margin < self.object_room_search_margin_m:
            margins.append(margin)
            margin *= 2.0
        margins.append(self.object_room_search_margin_m)
        for margin in sorted(set(round(value, 6) for value in margins)):
            ring_candidates = defaultdict(int)
            expanded_half_x = max(float(size[0]) * 0.5 + margin, margin)
            expanded_half_y = max(float(size[1]) * 0.5 + margin, margin)
            for index in range(8):
                ratio = float(index) / 7.0
                x = float(center[0]) - expanded_half_x + 2.0 * expanded_half_x * ratio
                y = float(center[1]) - expanded_half_y + 2.0 * expanded_half_y * ratio
                perimeter_points = (
                    (x, float(center[1]) - expanded_half_y),
                    (x, float(center[1]) + expanded_half_y),
                    (float(center[0]) - expanded_half_x, y),
                    (float(center[0]) + expanded_half_x, y),
                )
                for px, py in perimeter_points:
                    coords = world_to_grid(px, py, grid_info)
                    if coords is None:
                        continue
                    idx = grid_index(coords[0], coords[1], grid_info.width)
                    if 0 <= idx < len(scene_data):
                        room_id = int(scene_data[idx])
                        if room_id >= 0:
                            ring_candidates[room_id] += 1
            if ring_candidates:
                return max(
                    sorted(ring_candidates.keys()),
                    key=lambda room_id: ring_candidates[room_id],
                )
        return None

    def _infer_portal_room_ids(self, node):
        if not self.room_grid:
            return [node.room_id] if node.room_id is not None else []
        grid_info = self.room_grid["info"]
        scene_data = self.room_grid["scene_data"]
        if grid_info is None or not scene_data:
            return [node.room_id] if node.room_id is not None else []

        size = list(node.aabb_size or [])
        center = list(node.aabb_center or node.centroid or [])
        center.extend([0.0] * max(0, 3 - len(center)))
        size.extend([0.0] * max(0, 3 - len(size)))
        yaw_value = node.attributes.get("interaction_reference_yaw")
        if yaw_value is None:
            yaw_value = node.attributes.get("yaw")
        try:
            cache_yaw = float(yaw_value)
        except (TypeError, ValueError):
            cache_yaw = None
        cache_key = (
            int(self._room_grid_epoch),
            float(center[0]),
            float(center[1]),
            abs(float(size[0])),
            abs(float(size[1])),
            cache_yaw,
            self._resolve_room_id(node.room_id)
            if node.room_id is not None
            else None,
        )
        cached = self._portal_room_ids_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        def _finish(values):
            result = tuple(int(value) for value in values[:2])
            if len(self._portal_room_ids_cache) >= 512:
                self._portal_room_ids_cache.clear()
            self._portal_room_ids_cache[cache_key] = result
            return list(result)

        # A portal connects the free-space components on its two sides.  The
        # old implementation only counted a circular ring around the box;
        # when one room occupied most of that ring, the second room was lost
        # even though it was directly across the doorway.  Use the measured
        # OBB yaw (or the long AABB axis when yaw is unavailable) to cast
        # short rays through both doorway sides first.  This keeps the
        # association local and avoids accidentally linking a distant room.
        counts = defaultdict(int)
        center_x, center_y = float(center[0]), float(center[1])
        radii = sorted(
            {
                min(self.portal_room_max_radius_m, radius)
                for radius in (0.25, 0.50, 0.75, self.portal_room_max_radius_m)
                if radius > 0.0
            }
        )
        for radius_index, radius in enumerate(radii):
            weight = len(radii) - radius_index
            for index in range(48):
                angle = 2.0 * math.pi * float(index) / 48.0
                coords = world_to_grid(
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                    grid_info,
                )
                if coords is None:
                    continue
                data_index = grid_index(coords[0], coords[1], grid_info.width)
                if 0 <= data_index < len(scene_data):
                    room_id = int(scene_data[data_index])
                    if room_id >= 0:
                        counts[room_id] += weight

        # Read the portal's axial orientation.  ``interaction_reference_yaw``
        # is the stable OBB yaw; ``yaw`` is the live observation fallback.
        yaw = node.attributes.get("interaction_reference_yaw")
        if yaw is None:
            yaw = node.attributes.get("yaw")
        try:
            yaw = float(yaw)
            yaw_valid = math.isfinite(yaw)
        except (TypeError, ValueError):
            yaw_valid = False
        if not yaw_valid:
            size_x = abs(float(size[0]))
            size_y = abs(float(size[1]))
            # The long box axis is the doorway span, therefore its
            # perpendicular is the crossing direction.
            yaw = math.pi * 0.5 if size_y >= size_x else 0.0

        normal_x = -math.sin(yaw)
        normal_y = math.cos(yaw)
        tangent_x = math.cos(yaw)
        tangent_y = math.sin(yaw)
        span = max(
            abs(float(size[0])),
            abs(float(size[1])),
        )
        # Keep the side probe local to the doorway.  A few tangent offsets
        # make this robust to a slightly off-centre detector box without
        # widening the search enough to reach an unrelated room.
        tangent_offsets = (-0.25 * span, 0.0, 0.25 * span)
        side_room_ids = []
        for side in (-1.0, 1.0):
            side_counts = defaultdict(int)
            for radius_index, radius in enumerate(radii):
                # Near samples get more weight; farther samples are only a
                # fallback when the doorway opens into a larger room.
                weight = len(radii) - radius_index
                for offset in tangent_offsets:
                    px = center_x + side * normal_x * radius + tangent_x * offset
                    py = center_y + side * normal_y * radius + tangent_y * offset
                    coords = world_to_grid(px, py, grid_info)
                    if coords is None:
                        continue
                    data_index = grid_index(coords[0], coords[1], grid_info.width)
                    if 0 <= data_index < len(scene_data):
                        room_id = int(scene_data[data_index])
                        if room_id >= 0:
                            side_counts[room_id] += weight
            if side_counts:
                side_room_ids.append(
                    max(
                        sorted(side_counts.keys()),
                        key=lambda room_id: side_counts[room_id],
                    )
                )

        # Prefer two distinct room labels found on opposite sides.  Preserve
        # the current room as the first endpoint when possible so graph edge
        # ordering remains stable for navigation consumers.
        distinct_side_ids = []
        for room_id in side_room_ids:
            if room_id not in distinct_side_ids:
                distinct_side_ids.append(room_id)
        if len(distinct_side_ids) >= 2:
            current_room = (
                self._resolve_room_id(node.room_id)
                if node.room_id is not None
                else None
            )
            if current_room in distinct_side_ids:
                distinct_side_ids.remove(current_room)
                distinct_side_ids.insert(0, current_room)
            return _finish(distinct_side_ids)

        ranked = sorted(counts, key=lambda room_id: (-counts[room_id], room_id))
        return _finish(ranked)

    def _refresh_room_attributes(self):
        room_nodes = {
            int(node.room_id): node
            for node in self.nodes.values()
            if node.type == "room"
            and node.room_id is not None
            and node.attributes.get("active", True)
        }
        evidence_by_room = defaultdict(list)
        for node in self.nodes.values():
            if node.type in {"scene", "room", "portal"} or node.room_id is None:
                continue
            room_id = self._resolve_room_id(node.room_id)
            evidence_by_room[room_id].append(
                {
                    "node_id": node.id,
                    "semantic_name": node.label,
                    "category": node.attributes.get("category"),
                    "confidence": node.confidence,
                }
            )
        for room_id, room_node in room_nodes.items():
            result = self.room_attribute_inferencer.infer(
                evidence_by_room.get(room_id, [])
            )
            # Persist the deterministic prior separately.  It is useful for
            # diagnosis and remains the effective value while the MLLM lane is
            # pending, unavailable, stale, or low confidence.
            room_node.attributes.update(
                {
                    "room_attribute_rule": result["room_attribute"],
                    "room_attribute_rule_confidence": float(result["confidence"]),
                    "room_attribute_rule_scores": dict(result["scores"]),
                    "room_attribute_rule_evidence": list(result["evidence"]),
                }
            )
            mllm_attribute = normalize_label(
                room_node.attributes.get("room_mllm_attribute") or "unknown"
            )
            mllm_confidence = float(
                room_node.attributes.get("room_mllm_attribute_confidence", 0.0)
                or 0.0
            )
            mllm_is_usable = (
                room_node.attributes.get("room_attribute_status") == "ready"
                and mllm_attribute not in {"", "unknown"}
                and mllm_confidence >= self.room_mllm_min_confidence
            )
            # A room becomes a persistent semantic landmark only after two
            # accepted room-grid observations and a usable M1 room result.
            # Before that point it may still be shown as a provisional room,
            # but it remains eligible for the normal lifecycle handling.
            if (
                mllm_is_usable
                and int(room_node.attributes.get("room_observation_count", 0) or 0)
                >= 2
            ):
                room_node.attributes.update(
                    {
                        "persistent_semantic_node": True,
                        "semantic_confirmation": "two_room_frames_plus_m1",
                        "room_lifecycle": "persistent",
                        "active": True,
                    }
                )
            if mllm_is_usable:
                room_node.attributes.update(
                    {
                        "room_attribute": mllm_attribute,
                        "room_attribute_confidence": mllm_confidence,
                        "room_attribute_scores": dict(result["scores"]),
                        "room_attribute_evidence": list(
                            room_node.attributes.get(
                                "room_mllm_attribute_evidence_object_ids"
                            )
                            or []
                        ),
                        "room_attribute_source": str(
                            room_node.attributes.get("room_mllm_attribute_source")
                            or "mllm_room_attribute_inference"
                        ),
                    }
                )
            else:
                room_node.attributes.update(
                    {
                        "room_attribute": result["room_attribute"],
                        "room_attribute_confidence": float(result["confidence"]),
                        "room_attribute_scores": dict(result["scores"]),
                        "room_attribute_evidence": list(result["evidence"]),
                        "room_attribute_source": "weighted_object_types",
                    }
                )

    def _bump_revision(self):
        self.graph_revision += 1
        for node in self.nodes.values():
            node.graph_revision = self.graph_revision

    def _build_navigation_hints(self):
        hints = []
        counter = 1
        for node in sorted(self.nodes.values(), key=lambda item: item.id):
            if node.type == "room" and node.attributes.get("active", True):
                hints.append(
                    self._make_hint(counter, "room_center", node, False, node.id, "none", "known_room_center")
                )
                counter += 1
            elif node.type == "object":
                hints.append(
                    self._make_hint(counter, "target_object", node, False, None, "none", "detected_object")
                )
                counter += 1
            elif node.type == "portal":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_portal",
                        node,
                        bool(node.interaction.get("requires_interaction")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "portal_may_unlock_room",
                    )
                )
                counter += 1
            elif node.type == "container":
                hints.append(
                    self._make_hint(
                        counter,
                        "interactive_container",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "container_may_reveal_object",
                    )
                )
                counter += 1
            elif node.type == "support":
                hints.append(
                    self._make_hint(
                        counter,
                        "support_surface",
                        node,
                        bool(node.interaction.get("is_interactable")),
                        node.id,
                        node.interaction.get("interaction_mode", "none"),
                        "support_surface_context",
                    )
                )
                counter += 1
        return hints

    def _make_hint(self, index, hint_type, node, requires_interaction, interaction_node_id, interaction_mode, reason):
        return NavigationHint(
            hint_id=f"hint_{index:04d}",
            type=hint_type,
            node_id=node.id,
            position=list(node.centroid),
            room_id=node.room_id,
            priority=1.0,
            confidence=float(node.confidence),
            requires_interaction=bool(requires_interaction),
            interaction_node_id=interaction_node_id,
            interaction_mode=interaction_mode,
            state=node.interaction.get("state", "unknown"),
            reason=reason,
        )


def volume(size):
    return max(float(size[0]), 0.0) * max(float(size[1]), 0.0) * max(float(size[2]), 0.0)


def top_surface_z(node):
    return float(node.aabb_center[2]) + float(node.aabb_size[2]) * 0.5


def point_inside_2d(point, center, size, margin=0.05):
    half_x = float(size[0]) * 0.5 + margin
    half_y = float(size[1]) * 0.5 + margin
    return (
        abs(float(point[0]) - float(center[0])) <= half_x
        and abs(float(point[1]) - float(center[1])) <= half_y
    )


def point_inside_3d(point, center, size, margin=0.05):
    return (
        abs(float(point[0]) - float(center[0])) <= float(size[0]) * 0.5 + margin
        and abs(float(point[1]) - float(center[1])) <= float(size[1]) * 0.5 + margin
        and abs(float(point[2]) - float(center[2])) <= float(size[2]) * 0.5 + margin
    )


def _vertical_gap(obj, support):
    return abs(float(obj.centroid[2]) - top_surface_z(support))


def _support_height_limit(support):
    return max(0.25, float(support.aabb_size[2]) + 0.6)


def _same_room_or_unknown(obj, candidate):
    return candidate.room_id is None or obj.room_id is None or candidate.room_id == obj.room_id


def _support_xy_match(obj, support):
    return point_inside_2d(obj.centroid, support.aabb_center, support.aabb_size, margin=0.08)


def _object_above_support(obj, support):
    return float(obj.centroid[2]) >= top_surface_z(support) - 0.12


def _object_not_too_high(obj, support):
    return _vertical_gap(obj, support) <= _support_height_limit(support)


def _container_contains(obj, container, min_axis_fraction=0.90):
    for axis in range(3):
        object_center = float(obj.aabb_center[axis])
        object_size = max(0.0, float(obj.aabb_size[axis]))
        container_center = float(container.aabb_center[axis])
        container_size = max(0.0, float(container.aabb_size[axis]))
        container_min = container_center - 0.5 * container_size
        container_max = container_center + 0.5 * container_size
        if not container_min <= object_center <= container_max:
            return False
        if object_size <= 1e-8:
            continue
        object_min = object_center - 0.5 * object_size
        object_max = object_center + 0.5 * object_size
        overlap = max(0.0, min(object_max, container_max) - max(object_min, container_min))
        if overlap / object_size < float(min_axis_fraction):
            return False
    return True


def _is_plausible_container_content(obj, container):
    label = normalize_label(obj.label or obj.name)
    if any(token in label for token in ("plant", "flower", "tree")):
        return False
    container_volume = volume(container.aabb_size)
    object_volume = volume(obj.aabb_size)
    return container_volume > 1e-6 and object_volume <= min(0.10, 0.10 * container_volume)


# The physical detector can leave environmental fixtures in the same depth
# slab as an open refrigerator.  These labels are not contents even when their
# 3-D boxes are small enough to pass the relaxed open-door volume test.
_OPEN_REFRIGERATOR_NON_CONTENT_LABELS = (
    "person",
    "picture",
    "solar_battery",
    "shelve",
    "shelf",
    "rack",
    "waste",
    "grid",
    "logo",
    "chair",
    "watch",
    "safe",
    "plug",
    "broom",
    "faucet",
    "screen",
    "barber_shop",
    "locker",
    "plant",
    "houseplant",
    "succulent",
    "flower",
    "tree",
    "eucalyptus",
    "table",
    "desk",
    "bed",
    "sofa",
    "chair",
    "bench",
    "cabinet",
    "cupboard",
    "drawer",
    "wardrobe",
    "closet",
    "dresser",
    "microwave",
    "dishwasher",
    "oven",
    "stove",
    "sink",
    "door",
    "portal",
    "freezer",
    "fridge",
    "refrigerator",
    "wall",
    "floor",
    "ceiling",
)
_OPEN_REFRIGERATOR_CONTENT_LABELS = frozenset(
    {
        "food",
        "bottle",
        "cup",
        "bowl",
        "dairy",
        "lemon",
        "jug",
        "milk",
        "vinegar",
        "persimmon",
        "fruit",
        "vegetable",
        "sushi",
        "can",
        "carton",
        "jar",
        "plate",
    }
)


def _valid_float_scalar(value):
    """Return whether *value* is a finite scalar suitable for geometry."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _valid_float_vector(value, length):
    """Return whether a sequence has ``length`` finite numeric entries."""

    if not isinstance(value, (list, tuple)) or len(value) < int(length):
        return False
    try:
        return all(math.isfinite(float(value[index])) for index in range(length))
    except (IndexError, TypeError, ValueError):
        return False


def _is_refrigerator_label(value):
    """Match refrigerator-family labels without treating arbitrary IDs as one."""

    label = normalize_label(value)
    cached = _REFRIGERATOR_LABEL_CACHE.get(label)
    if cached is not None:
        return cached
    result = any(
        _cached_label_marker_match(label, marker)
        for marker in ("fridge", "freezer", "refrigerator")
    )
    if len(_REFRIGERATOR_LABEL_CACHE) >= _LABEL_MARKER_CACHE_MAX:
        _REFRIGERATOR_LABEL_CACHE.clear()
    _REFRIGERATOR_LABEL_CACHE[label] = result
    return result


def _is_refrigerator_content_label(label):
    """Return whether a label describes a likely movable fridge item."""

    return any(
        _label_matches_marker(label, marker)
        for marker in _OPEN_REFRIGERATOR_CONTENT_LABELS
    )


def _source_node_type(node):
    """Return the topology type supplied by the geometry/source lane.

    ``node.type`` is intentionally mutable because a confident M1 response
    may refine an object to a container.  The source type is the stable
    provenance needed by the refrigerator relation fallback.
    """

    attributes = node.attributes or {}
    value = attributes.get("topology_type") or attributes.get(
        "observation_node_type"
    )
    return str(value or "").strip().casefold()


def _source_node_labels(node):
    """Return only labels owned by the source observation.

    M1's ``observed_object_name`` and its copied ``semantic_name``/``category``
    are deliberately excluded.  In the physical run a bottle was once
    answered as "refrigerator" by M1; treating that transient answer as a
    negative refrigerator-content marker was enough to hide the real child.
    ``candidate_labels`` are also omitted here because they are an open-vocab
    hypothesis list, not a primary source class.
    """

    attributes = node.attributes or {}
    values = [
        attributes.get("source_semantic_name"),
        attributes.get("source_category"),
    ]
    # Legacy/replay nodes created before source provenance was persisted still
    # need the old behavior.  If M1 has not overridden the name, the public
    # fields are a safe fallback; otherwise no mutable M1 label is promoted to
    # source evidence.
    if not any(normalize_label(value) for value in values):
        if not bool(attributes.get("m1_name_override")):
            values.extend(
                [
                    node.label,
                    node.name,
                    attributes.get("semantic_name"),
                    attributes.get("category"),
                ]
            )
    return tuple(
        sorted(
            {
                normalized
                for normalized in (normalize_label(value) for value in values)
                if normalized
            }
        )
    )


def _source_node_has_label(node, predicate):
    """Return whether a predicate matches at least one source label."""

    return any(predicate(label) for label in _source_node_labels(node))


def _looks_like_refrigerator_body(node):
    """Reject tiny M1-only refrigerator hallucinations as parent containers."""

    size = _open_refrigerator_reference_size(node)
    if size is None or len(size) < 3:
        return False
    try:
        horizontal = sorted(
            [abs(float(size[0])), abs(float(size[1]))], reverse=True
        )
        vertical = abs(float(size[2]))
        body_volume = volume(size)
    except (IndexError, TypeError, ValueError):
        return False
    # This is intentionally permissive for a mini-fridge, but far above the
    # small crop boxes produced for a bottle/hand-held item.
    return bool(
        horizontal[0] >= 0.35
        and vertical >= 0.65
        and body_volume >= 0.18
    )


def _has_confirmed_refrigerator_interaction(node):
    """Return whether the node has a sealed open/close interaction fact."""

    interaction = node.interaction or {}
    for event in reversed(interaction.get("operation_history") or []):
        if not isinstance(event, dict) or not bool(event.get("success")):
            continue
        action = str(event.get("action") or "").strip().casefold()
        post_state = str(event.get("post_state") or "").strip().casefold()
        if action in {"open", "close", "open_close", "opened", "closed"}:
            return True
        if post_state in {
            "open",
            "opened",
            "ajar",
            "closed",
            "static_open",
            "static_closed",
        }:
            return True
    return False


def _restore_confirmed_refrigerator_container_type(node):
    """Keep a proven refrigerator in the container relation/candidate pool.

    M1 class overrides are intentionally authoritative before an interaction
    has happened.  Once the physical bridge has returned a successful
    open/close result, however, a later one-frame ``object`` answer must not
    remove the appliance from the graph just as its contents become visible.
    This narrow repair requires source-owned refrigerator evidence and a
    body-sized geometry, so it cannot resurrect the small false-positive
    crops that the candidate guard rejects.
    """

    if node is None or str(node.type or "").casefold() == "container":
        return False
    attributes = node.attributes or {}
    source_labels = _source_node_labels(node)
    reference_label = attributes.get("interaction_reference_label")
    m1_reference_label = attributes.get("m1_observed_object_name")
    source_is_refrigerator = any(
        _is_refrigerator_label(label) for label in source_labels
    )
    refrigerator_identity = source_is_refrigerator or _is_refrigerator_label(
        reference_label
    ) or (
        bool(attributes.get("m1_refrigerator_evidence"))
        and _is_refrigerator_label(m1_reference_label)
    )
    if not refrigerator_identity:
        return False
    # A large/noisy bottle crop can occasionally pass the permissive body
    # dimensions.  Source-owned content labels still veto a mutable M1 fridge
    # promotion unless the detector itself identified the appliance.
    if not source_is_refrigerator and any(
        _is_refrigerator_content_label(label) for label in source_labels
    ):
        return False
    source_type = _source_node_type(node)
    if source_type in {"portal", "support", "scene", "room"}:
        return False
    if not _looks_like_refrigerator_body(node):
        return False
    if not _has_confirmed_refrigerator_interaction(node):
        return False
    node.type = "container"
    attributes["refrigerator_type_restored_after_success"] = True
    attributes["refrigerator_type_restore_source"] = "successful_interaction_history"
    return True


def _node_labels(node):
    """Return normalized labels available on a graph node."""

    attributes = node.attributes or {}
    values = [
        node.label,
        node.name,
        attributes.get("semantic_name"),
        attributes.get("category"),
        attributes.get("interaction_reference_label"),
        attributes.get("m1_observed_object_name"),
    ]
    # ``candidate_labels`` are useful only when they have actual tracker
    # support.  A raw open-vocabulary candidate list can contain unrelated
    # words, so include only labels with a non-trivial vote (or the first few
    # labels when the producer did not send votes at all).
    candidate_labels = [
        normalize_label(value) for value in attributes.get("candidate_labels") or []
    ]
    votes = attributes.get("label_votes") or {}
    if isinstance(votes, dict) and votes:
        try:
            max_vote = max(float(score) for score in votes.values())
        except (TypeError, ValueError):
            max_vote = 0.0
        for label in candidate_labels:
            try:
                score = float(votes.get(label, votes.get(str(label), 0.0)) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score >= max(0.05, 0.20 * max_vote):
                values.append(label)
    else:
        values.extend(candidate_labels[:4])
    return tuple(sorted({label for label in values if label}))


def _label_matches_marker(label, marker):
    """Match a semantic marker without treating ``track_*`` IDs as labels."""

    label = normalize_label(label)
    marker = normalize_label(marker)
    return _cached_label_marker_match(label, marker)


def _is_open_refrigerator_content_candidate(node):
    """Filter graph nodes before the relaxed open-door relation pass."""

    # Negative appliance/furniture markers must come from the source lane.
    # M1's mutable name is allowed as positive evidence (for example, a
    # source ``object`` crop refined to ``bottle``), but a one-frame M1
    # "refrigerator" answer must not erase a source bottle from this pool.
    labels = _node_labels(node)
    source_labels = _source_node_labels(node)
    if any(
        _label_matches_marker(label, token)
        for label in source_labels
        for token in _OPEN_REFRIGERATOR_NON_CONTENT_LABELS
    ):
        return False
    source_type = _source_node_type(node)
    if source_type in {"scene", "room", "portal", "support"}:
        return False
    # A source refrigerator/appliance is never its own content, even if M1
    # temporarily changed its public graph type to ``object``.
    if _source_node_has_label(node, _is_refrigerator_label):
        return False
    # Source-observed objects are already eligible.  A node that M1 relabeled
    # as a container is eligible only when its semantic label still identifies
    # a likely item (or is an explicitly generic object token); this avoids
    # recursively assigning cabinets, boxes, and appliances as contents.
    if source_type == "object":
        return True
    if str(node.type or "").casefold() != "container":
        return True
    generic_labels = {"object", "item", "thing", "unknown", "container"}
    return any(
        label in generic_labels or _is_refrigerator_content_label(label)
        for label in labels
    )


def _is_open_refrigerator(container):
    """Whether a container has a confirmed open refrigerator state."""

    labels = _node_labels(container)
    source_labels = _source_node_labels(container)
    source_type = _source_node_type(container)
    source_is_refrigerator = any(
        _is_refrigerator_label(label) for label in source_labels
    )
    # A stable refrigerator reference can be created from a confident M1
    # answer on a generic crop.  It must not override a source-owned bottle/
    # food label, even if a stale command result later lands on that track.
    # Otherwise the false track could become an ``open`` parent and absorb
    # nearby objects during the same relation rebuild.
    if not source_is_refrigerator and any(
        _is_refrigerator_content_label(label) for label in source_labels
    ):
        return False
    stable_reference_is_refrigerator = _is_refrigerator_label(
        (container.attributes or {}).get("interaction_reference_label")
    )
    m1_only_refrigerator = any(_is_refrigerator_label(label) for label in labels)
    if not (source_is_refrigerator or stable_reference_is_refrigerator):
        # Permit a genuinely large source crop that M1 promoted to a
        # refrigerator, but never let a tiny/object-like M1 hallucination
        # become a parent container.  This keeps M1 useful when the detector
        # emitted only a generic object while preserving source bottle tracks.
        if not m1_only_refrigerator or not _looks_like_refrigerator_body(container):
            return False
        if source_type == "object" and any(
            _is_refrigerator_content_label(label) for label in source_labels
        ):
            return False
    if source_type in {"portal", "support", "scene", "room"}:
        return False
    interaction = container.interaction or {}
    attributes = container.attributes or {}
    override = attributes.get("interaction_state_override") or {}

    state = str(
        interaction.get("state")
        or override.get("state")
        or interaction.get("coarse_state")
        or override.get("coarse_state")
        or ""
    ).strip().casefold()
    capability = str(
        interaction.get("capability")
        or override.get("capability")
        or ""
    ).strip().casefold()
    state_source = str(
        interaction.get("state_source")
        or override.get("state_source")
        or ""
    ).strip().casefold()

    # A later successful close/blocked transition supersedes an earlier open
    # event.  Otherwise stale history would keep projecting contents after a
    # refrigerator had been closed again.
    history = interaction.get("operation_history") or []
    latest_successful_transition = None
    for event in reversed(history):
        if not isinstance(event, dict) or not bool(event.get("success")):
            continue
        action = str(event.get("action") or "").strip().casefold()
        post_state = str(event.get("post_state") or "").strip().casefold()
        if (
            action in {"close", "closed"}
            or post_state in {"closed", "static_closed", "blocked", "unavailable"}
        ):
            latest_successful_transition = "closed"
            break
        if (
            post_state in {"open", "opened", "ajar", "static_open"}
            or action in {"open", "open_close"}
        ):
            latest_successful_transition = "open"
            break
    if latest_successful_transition == "closed" or state in {
        "closed",
        "static_closed",
        "blocked",
        "unavailable",
    }:
        return False

    # A visual M1 answer is deliberately not enough to expose contents.  The
    # physical bridge may be asynchronous, and accepting that one frame here
    # made a detector hallucination look like a completed fridge action.  A
    # successful operation history entry (or an explicitly static-open
    # executor/oracle result) is the trusted state transition.
    trusted_static = state == "static_open" and capability == "static"
    # Preserve compatibility with replay/oracle fixtures that provide an
    # explicit open state but no operation history, while excluding mllm-only
    # visual patches.  The source check is intentionally narrow.
    trusted_explicit_open = state in {"open", "opened", "ajar"} and (
        state_source in {
            "successful_action_postcondition",
            "executor_feedback",
            "oracle_interaction",
            "direct_joint_readback",
            "interaction_result",
            "evaluator_object_skill",
        }
        or (
            "mllm" not in state_source
            and interaction.get("capability_observed_step") is not None
        )
    )
    return bool(
        latest_successful_transition == "open"
        or trusted_static
        or trusted_explicit_open
    )


def _node_yaw(node):
    """Read a detector OBB yaw, falling back to an axis-aligned box."""

    attributes = node.attributes or {}
    for key in ("interaction_reference_yaw", "yaw"):
        value = attributes.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return 0.0


def _open_refrigerator_reference_center(container):
    """Use a stable interaction reference center when one is available."""

    attributes = container.attributes or {}
    values = attributes.get("interaction_reference_aabb_center")
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        values = container.aabb_center
    try:
        return [float(values[index]) for index in range(3)]
    except (IndexError, TypeError, ValueError):
        return None


def _open_refrigerator_reference_size(container):
    """Return the stable body-box dimensions used by the depth corridor."""

    attributes = container.attributes or {}
    values = attributes.get("interaction_reference_aabb_size")
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        values = container.aabb_size
    try:
        return [max(0.0, float(values[index])) for index in range(3)]
    except (IndexError, TypeError, ValueError):
        return None


def _open_refrigerator_depth_lateral_axes(container):
    """Return ``(depth_axis, lateral_axis)`` in the refrigerator OBB frame.

    Refrigerator bodies are usually deeper along their shorter horizontal
    OBB dimension.  Choosing that axis instead of hard-coding local ``y``
    keeps the inference valid when the detector swaps the two OBB axes.
    """

    size = _open_refrigerator_reference_size(container)
    if size is None or len(size) < 2:
        return 1, 0
    try:
        # Tie-breaking to axis 1 preserves the historical local-x/local-y
        # convention used by existing observations.
        depth_axis = 0 if float(size[0]) < float(size[1]) - 1e-6 else 1
    except (TypeError, ValueError):
        depth_axis = 1
    return depth_axis, 1 - depth_axis


def _open_refrigerator_local_xy(obj, container):
    """Transform an object's world center into the refrigerator OBB frame."""

    center = _open_refrigerator_reference_center(container)
    if center is None:
        return None
    try:
        dx = float(obj.aabb_center[0]) - center[0]
        dy = float(obj.aabb_center[1]) - center[1]
    except (IndexError, TypeError, ValueError):
        return None
    yaw = _node_yaw(container)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
    )


def _open_refrigerator_geometry_candidate(obj, container):
    """Return local coordinates when an object plausibly lies in an open door.

    The detector's box for an open appliance is frequently the *closed* body
    box, while contents are projected in front of it.  Use conservative
    lateral/vertical/volume bounds and a bounded depth extension so nearby
    room fixtures are not swept into the container.
    """

    if not _same_room_or_unknown(obj, container):
        return None
    # Use source-owned labels for exclusion.  A mutable M1 name such as
    # ``refrigerator`` can be wrong for an otherwise valid bottle/content
    # track; geometry and source provenance should remain authoritative for
    # this negative filter.
    labels = _source_node_labels(obj)
    if any(
        _label_matches_marker(label, token)
        for label in labels
        for token in _OPEN_REFRIGERATOR_NON_CONTENT_LABELS
    ):
        return None
    reference_size = _open_refrigerator_reference_size(container)
    if reference_size is None:
        return None
    container_volume = volume(reference_size)
    object_volume = volume(obj.aabb_size)
    if container_volume <= 1e-6:
        return None
    # Keep a larger allowance than the strict closed-volume rule for noisy
    # bottle/box detections, but never admit furniture-sized observations.
    if object_volume > max(0.35, 0.35 * container_volume):
        return None
    try:
        reference_center = _open_refrigerator_reference_center(container)
        if reference_center is None:
            return None
        object_z = float(obj.aabb_center[2])
        object_half_z = max(0.0, float(obj.aabb_size[2])) * 0.5
        container_z = reference_center[2]
        container_half_z = max(0.0, float(reference_size[2])) * 0.5
        object_half_x = max(0.0, float(obj.aabb_size[0])) * 0.5
        object_half_y = max(0.0, float(obj.aabb_size[1])) * 0.5
        container_half_x = max(0.0, float(reference_size[0])) * 0.5
        container_half_y = max(0.0, float(reference_size[1])) * 0.5
    except (IndexError, TypeError, ValueError):
        return None
    # Contents may be detected a little below the body bottom (e.g. a bowl on
    # the lowest shelf), but not several metres above/below the appliance.
    vertical_margin = 0.20
    if (
        object_z + object_half_z < container_z - container_half_z - vertical_margin
        or object_z - object_half_z > container_z + container_half_z + vertical_margin
    ):
        return None
    local_xy = _open_refrigerator_local_xy(obj, container)
    if local_xy is None:
        return None
    depth_axis, lateral_axis = _open_refrigerator_depth_lateral_axes(container)
    lateral = local_xy[lateral_axis]
    depth = local_xy[depth_axis]
    # The larger horizontal extent is the appliance opening width.  Cap the
    # contribution of a noisy object box so one oversized bottle does not get
    # rejected solely because its depth box is inflated.
    opening_half_width = (container_half_x, container_half_y)[lateral_axis]
    object_horizontal_half = min(max(object_half_x, object_half_y), 0.18)
    if abs(lateral) + object_horizontal_half > opening_half_width + 0.12:
        return None
    body_half_depth = (container_half_x, container_half_y)[depth_axis]
    minimum_depth = max(0.05, body_half_depth - 0.12)
    maximum_depth = body_half_depth + max(0.90, 1.5 * body_half_depth)
    if abs(depth) < minimum_depth or abs(depth) > maximum_depth:
        return None
    # Downstream side selection intentionally receives a stable
    # ``(lateral, depth)`` pair regardless of which OBB axis is deeper.
    return lateral, depth


def _open_refrigerator_content_score(obj, container, side):
    """Score an open-door match so the nearest compatible fridge wins."""

    local_candidate = _open_refrigerator_geometry_candidate(obj, container)
    if local_candidate is None:
        return float("inf")
    lateral, depth = local_candidate
    reference_size = _open_refrigerator_reference_size(container) or [0.0, 0.0, 0.0]
    depth_axis, lateral_axis = _open_refrigerator_depth_lateral_axes(container)
    try:
        body_half_depth = max(0.01, float(reference_size[depth_axis]) * 0.5)
        lateral_half_width = max(0.01, float(reference_size[lateral_axis]) * 0.5)
    except (IndexError, TypeError, ValueError):
        body_half_depth = 0.01
        lateral_half_width = 0.01
    # Prefer items close to the exposed face and near the opening centre.  The
    # side term is intentionally a hard filter in the caller; this score only
    # resolves multiple compatible refrigerators or noisy duplicate tracks.
    depth_distance = abs(abs(float(depth)) - body_half_depth)
    lateral_distance = abs(float(lateral)) / lateral_half_width
    score = depth_distance + 0.08 * lateral_distance
    labels = _node_labels(obj)
    if any(_is_refrigerator_content_label(label) for label in labels):
        score -= 0.05
    if float(depth) * float(side) < 0.0:
        score += 0.25
    return max(0.0, score)


def _open_refrigerator_approach_side(container):
    """Return the local depth side from which the successful open was viewed."""

    center = _open_refrigerator_reference_center(container) or []
    if len(center) < 2:
        return None
    approaches = []
    history = (container.interaction or {}).get("operation_history") or []
    for event in reversed(history):
        if not isinstance(event, dict) or not bool(event.get("success")):
            continue
        action = str(event.get("action") or "").strip().casefold()
        if action and action not in {"open", "open_close"}:
            continue
        approach = list(event.get("approach_goal_xyyaw") or [])
        if len(approach) >= 2:
            approaches.append(approach)
    attributes = container.attributes or {}
    configured_approach = list(
        attributes.get("interaction_approach_pose_xyyaw") or []
    )
    if len(configured_approach) >= 2:
        approaches.append(configured_approach)
    yaw = _node_yaw(container)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    depth_axis, _ = _open_refrigerator_depth_lateral_axes(container)
    for approach in approaches:
        try:
            # Contents are exposed toward the robot/camera approach point,
            # i.e. from the appliance center toward the approach pose.
            dx = float(approach[0]) - float(center[0])
            dy = float(approach[1]) - float(center[1])
        except (TypeError, ValueError):
            continue
        local_xy = (
            cos_yaw * dx + sin_yaw * dy,
            -sin_yaw * dx + cos_yaw * dy,
        )
        local_depth = local_xy[depth_axis]
        if abs(local_depth) > 0.15 and math.isfinite(local_depth):
            return 1 if local_depth > 0.0 else -1
    return None


def _infer_open_refrigerator_content_side(container, object_nodes):
    """Infer the populated side of an open refrigerator in local depth."""

    scores = {-1: 0.0, 1: 0.0}
    for obj in object_nodes:
        if not _is_open_refrigerator_content_candidate(obj):
            continue
        local_xy = _open_refrigerator_geometry_candidate(obj, container)
        if local_xy is None or abs(local_xy[1]) <= 1e-6:
            continue
        # A semantic food/container token is useful evidence, but geometry is
        # sufficient so unknown detector labels remain mappable.
        labels = _node_labels(obj)
        weight = 1.0 + (
            0.15
            if any(_is_refrigerator_content_label(label) for label in labels)
            else 0.0
        )
        scores[1 if local_xy[1] > 0.0 else -1] += weight
    positive, negative = scores[1], scores[-1]
    if positive <= 0.0 and negative <= 0.0:
        return None, 0.0
    previous = str(
        (container.attributes or {}).get("open_refrigerator_content_side") or ""
    ).strip().casefold()
    previous_side = (
        1
        if previous in {"positive_depth", "positive", "+1", "1"}
        else -1
        if previous in {"negative_depth", "negative", "-1"}
        else None
    )
    approach_side = _open_refrigerator_approach_side(container)
    if approach_side is not None and scores[approach_side] > 0.0:
        # The approach pose is a causal view of the opened face.  Prefer it
        # whenever there is candidate evidence on that side; this prevents a
        # nearby small fixture on the opposite side from winning a raw count
        # tie, while still falling back to geometry when the pose is absent.
        side = approach_side
    elif abs(positive - negative) <= 1e-6 and previous_side is not None:
        side = previous_side
    else:
        side = 1 if positive >= negative else -1
    total = positive + negative
    confidence = abs(positive - negative) / total if total > 1e-6 else 0.0
    return side, confidence


def _is_open_refrigerator_content(obj, container, side):
    """Check an object against the selected open-refrigerator depth side."""

    local_xy = _open_refrigerator_geometry_candidate(obj, container)
    if local_xy is None:
        return False
    return local_xy[1] * float(side) > 0.0


InteractionGraphStore._is_inside_volume = staticmethod(_container_contains)
InteractionGraphStore._is_on_support = staticmethod(
    lambda obj, support: _same_room_or_unknown(obj, support)
    and _support_xy_match(obj, support)
    and _object_above_support(obj, support)
    and _object_not_too_high(obj, support)
)
