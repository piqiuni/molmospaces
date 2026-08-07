from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .geometry_utils import normalize_label

PORTAL_LABELS = {
    "portal",
    "door",
    "doorframe",
    "doorway",
    "door_leaf",
    "gate",
    "entrance",
}
SUPPORT_LABELS = {
    "table",
    "desk",
    "bed",
    "sofa",
    "couch",
    "nightstand",
    "countertop",
    "counter",
    "shelf",
    "rack",
    "bench",
    "cabinet_top",
}
CONTAINER_LABELS = {
    "fridge",
    "refrigerator",
    "cabinet",
    "drawer",
    "wardrobe",
    "closet",
    "cupboard",
    "dresser",
    "chest_of_drawers",
    "microwave",
    "dishwasher",
    "box",
    "storage_bin",
}
OPENABLE_CONTAINER_LABELS = CONTAINER_LABELS - {"box", "storage_bin"}

HINGE_NAMES = {"hinge", "mjjnthinge"}
SLIDE_NAMES = {"slide", "mjJNT_SLIDE", "mjjntslide"}

BOX_ONLY_PERCEPTION_CONTRACT = "exact_2d_3d_boxes_only"
# Restricted-GT doors use one deliberately boring public identity.  The
# suffix is an episode-local ordinal allocated by the publisher (``door_0001``
# etc.); ``door_x`` is also accepted as a compact fixture token.  Never expose
# a source/body name such as ``doorframe_static_17`` on the public wire.
_PUBLIC_DOOR_ID_RE = re.compile(r"^door_(?:[0-9]{1,8}|x)$")


def is_public_door_id(value: Any) -> bool:
    """Return whether *value* is one of our generic public door references."""

    return bool(_PUBLIC_DOOR_ID_RE.fullmatch(str(value or "").strip().casefold()))


def sanitize_token(value: str) -> str:
    text = normalize_label(value)
    return text.replace("/", "_").replace("|", "_").replace(":", "_")


def opaque_door_instance_id(value: Any) -> str:
    """Return a generic, stable door identity for legacy/raw observations.

    Live restricted-GT observations already carry the publisher-assigned
    ordinal.  This fallback is used by older producers and unit fixtures.  It
    hashes a canonicalized alias token into a numeric suffix, so ``doorframe``,
    ``doorway`` and ``door_leaf`` variants of the same source token converge
    without putting the source text on the wire.
    """

    raw = str(value or "").strip().casefold()
    if is_public_door_id(raw):
        return raw
    canonical = normalize_label(raw)
    # These are geometry/asset aliases, not distinct semantic objects.  Remove
    # them before hashing so frame/leaf/doorway names receive the same public
    # identity when an old producer sends more than one alias.
    canonical = re.sub(
        r"(?:^|_)(?:doorframe|doorway|door_leaf|door|portal)(?=_|$)",
        "_",
        canonical,
    )
    canonical = re.sub(r"_+", "_", canonical).strip("_") or "door"
    digest = hashlib.blake2s(canonical.encode("utf-8"), digest_size=4).digest()
    ordinal = int.from_bytes(digest, "big") % 9_999 + 1
    return f"door_{ordinal:04d}"


# Keep the old helper name as a source-compatible alias for downstream tools;
# its return value now follows the generic ``door_<ordinal>`` contract.
def opaque_portal_instance_id(value: Any) -> str:
    return opaque_door_instance_id(value)


def point3(values=None):
    if isinstance(values, dict):
        return [
            float(values.get("x", 0.0)),
            float(values.get("y", 0.0)),
            float(values.get("z", 0.0)),
        ]
    vals = list(values or [])
    if len(vals) < 3:
        vals.extend([0.0] * (3 - len(vals)))
    return [float(vals[0]), float(vals[1]), float(vals[2])]


def unit_axis_xy(values: Any) -> list[float]:
    """Normalize an explicit interaction normal without accepting bad input."""

    raw = list(values or [])
    if len(raw) < 2:
        return []
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return []
    norm = math.hypot(x, y)
    if not math.isfinite(norm) or norm <= 1e-6:
        return []
    return [x / norm, y / norm]


def segmentation_pixel_count(segmentation: Any) -> int:
    if isinstance(segmentation, dict):
        rle_count = rle_foreground_pixel_count(segmentation)
        if rle_count is not None:
            return rle_count
        rows = list(segmentation.get("rows") or [])
        cols = list(segmentation.get("cols") or [])
        return min(len(rows), len(cols))
    if isinstance(segmentation, list):
        return sum(
            1
            for row in segmentation
            if isinstance(row, list)
            for value in row
            if bool(value)
        )
    return 0


def rle_foreground_pixel_count(mask_rle: Any) -> int | None:
    """Return the foreground-pixel count of an uncompressed binary mask RLE.

    Restricted-GT observations use the standard alternating background/foreground
    run convention, beginning with a background run.  Counting the odd runs is
    sufficient for visibility gating and avoids materialising a full-resolution
    mask in the semantic-mapping process.

    ``None`` denotes a non-RLE payload so callers can retain support for the
    legacy sparse ``{"rows": ..., "cols": ...}`` representation.  Malformed
    RLE is treated as zero visible pixels instead of trusting an invalid count.
    """

    if not isinstance(mask_rle, dict) or "counts" not in mask_rle:
        return None
    size = mask_rle.get("size")
    counts = mask_rle.get("counts")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or isinstance(counts, (str, bytes))
    ):
        return 0
    try:
        height = int(size[0])
        width = int(size[1])
    except (TypeError, ValueError):
        return 0
    if height < 0 or width < 0:
        return 0

    total = 0
    foreground = 0
    try:
        for index, value in enumerate(counts):
            run = int(value)
            if run < 0:
                return 0
            total += run
            if index % 2 == 1:
                foreground += run
    except (TypeError, ValueError):
        return 0
    if total != height * width:
        return 0
    return foreground


def bbox_area(bbox: Any) -> float:
    values = list(bbox or [])
    if len(values) < 4:
        return 0.0
    return max(0.0, float(values[2]) - float(values[0]) + 1.0) * max(
        0.0, float(values[3]) - float(values[1]) + 1.0
    )


def normalize_joint_type(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    lowered = text.lower()
    if lowered in HINGE_NAMES or "hinge" in lowered:
        return "hinge"
    if lowered in {name.lower() for name in SLIDE_NAMES} or "slide" in lowered:
        return "slide"
    return "none"


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    minimal_gt = bool(
        observation.get("id") is not None and observation.get("box_3d") is not None
    )
    box_3d = observation.get("box_3d") or {}
    if not isinstance(box_3d, dict):
        box_3d = {}
    if minimal_gt:
        raw_semantic_name = normalize_label(observation.get("name"))
        raw_instance_id = str(observation.get("id") or "")
        is_portal = raw_semantic_name in PORTAL_LABELS or is_public_door_id(raw_instance_id)
        # A GT ``doorframe`` / ``doorway`` / ``door_leaf`` is an asset
        # annotation.  Keep one generic door identity and the internal portal
        # topology class on the public wire contract.
        semantic_name = "portal" if is_portal else raw_semantic_name
        category = "door" if is_portal else semantic_name or "object"
        instance_id = (
            raw_instance_id
            if is_public_door_id(raw_instance_id)
            else opaque_door_instance_id(raw_instance_id)
            if is_portal
            else raw_instance_id
        )
        private_instance_id = raw_instance_id if is_portal else ""
        position = point3(box_3d.get("center"))
        aabb_center = point3(box_3d.get("center"))
        aabb_size = point3(box_3d.get("size"))
        bbox_2d = list(observation.get("bbox_2d") or [])
        segmentation = observation.get("mask_rle")
        if segmentation is None:
            segmentation = observation.get("segmentation_rle")
        if segmentation is None:
            segmentation = observation.get("segmentation")
        if segmentation is not None:
            visible_pixels = segmentation_pixel_count(segmentation)
            area = bbox_area(bbox_2d)
            visible_fraction = (
                min(1.0, float(visible_pixels) / area) if area > 0.0 else 0.0
            )
        else:
            visible_pixels = int(observation.get("visible_pixels", 0) or 0)
            visible_fraction = observation.get("visible_fraction")
            if visible_fraction is None:
                area = bbox_area(bbox_2d)
                visible_fraction = (
                    min(1.0, float(visible_pixels) / area) if area > 0.0 else 0.0
                )
        # Normal restricted-GT messages remain axis-free.  A rule-only run can
        # opt in to a live joint-derived normal without loading scene poses.
        oracle_rule_gt_axis = bool(
            observation.get("oracle_rule_gt_interaction_axis", False)
        )
        interaction_approach_axis_xy = (
            unit_axis_xy(observation.get("interaction_approach_axis_xy"))
            if oracle_rule_gt_axis
            else []
        )
        interaction_approach_axis_source = (
            str(observation.get("interaction_approach_axis_source") or "")
            if interaction_approach_axis_xy
            else ""
        )
    else:
        semantic_name = normalize_label(
            observation.get("semantic_name")
            or observation.get("semantic_class")
            or observation.get("class")
            or observation.get("name")
        )
        # Detector aliases such as ``doorframe`` are normalized into the one
        # public portal class as well.  Their original text is not retained in
        # node labels/names or MLLM context.
        if semantic_name in PORTAL_LABELS:
            semantic_name = "portal"
        category = (
            "portal"
            if semantic_name == "portal"
            else str(observation.get("category") or semantic_name or "object")
        )
        instance_id = str(observation.get("instance_id") or observation.get("id") or "")
        private_instance_id = ""
        position = point3(
            observation.get("position")
            or observation.get("coord")
            or observation.get("centroid")
            or box_3d.get("center")
        )
        aabb_center = point3(
            observation.get("aabb_center")
            or observation.get("box3d_center")
            or box_3d.get("center")
            or position
        )
        aabb_size = point3(
            observation.get("aabb_size")
            or observation.get("size")
            or observation.get("box3d_size")
            or box_3d.get("size")
        )
        bbox_2d = list(
            observation.get("bbox_2d") or observation.get("bbox") or []
        )
        segmentation = observation.get("segmentation")
        if segmentation is None:
            segmentation = observation.get("mask")
        visible_pixels = int(
            observation.get("visible_pixels", segmentation_pixel_count(segmentation))
            or 0
        )
        visible_fraction = observation.get("visible_fraction")
        if visible_fraction is None:
            area = bbox_area(bbox_2d)
            visible_fraction = (
                min(1.0, float(visible_pixels) / area) if area > 0.0 else 0.0
            )
        interaction_approach_axis_xy = unit_axis_xy(
            observation.get("interaction_approach_axis_xy")
        )
        interaction_approach_axis_source = str(
            observation.get("interaction_approach_axis_source") or ""
        )
    connected_room_ids = [] if minimal_gt else observation.get("connected_room_ids") or []
    room_id = None if minimal_gt else observation.get("room_id")

    if room_id is not None:
        try:
            room_id = int(room_id)
        except (TypeError, ValueError):
            room_id = None
    viz_aabb_center = (
        box_3d.get("center")
        if minimal_gt
        else observation.get("viz_aabb_center")
        or observation.get("world_box3d_center")
        or observation.get("aabb_center")
    )
    viz_aabb_size = (
        box_3d.get("size")
        if minimal_gt
        else observation.get("viz_aabb_size")
        or observation.get("world_box3d_size")
        or observation.get("aabb_size")
        or observation.get("size")
    )
    return {
        "minimal_gt_observation": minimal_gt,
        "observation_id": str(
            instance_id
            if minimal_gt
            else observation.get("observation_id") or observation.get("id") or ""
        ),
        "instance_id": instance_id,
        "private_instance_id": private_instance_id,
        "semantic_name": semantic_name or "object",
        "category": category,
        "candidate_labels": [] if minimal_gt else list(observation.get("candidate_labels") or []),
        "label_votes": {} if minimal_gt else dict(observation.get("label_votes") or {}),
        "confidence": 1.0 if minimal_gt else float(
            observation.get("confidence", observation.get("conf", 0.0)) or 0.0

        ),
        "position": position,
        "aabb_center": aabb_center,
        "aabb_size": aabb_size,
        "room_id": room_id,
        "connected_room_ids": [int(room) for room in connected_room_ids if room is not None],
        "parent": None if minimal_gt else observation.get("parent"),
        "children": [] if minimal_gt else list(observation.get("children") or []),
        "is_receptacle": False if minimal_gt else bool(observation.get("is_receptacle", False)),
        "is_pickup_candidate": False if minimal_gt else bool(observation.get("is_pickup_candidate", False)),
        "is_articulable": False if minimal_gt else bool(observation.get("is_articulable", False)),
        "is_door": (
            semantic_name == "portal"
            if minimal_gt
            else bool(observation.get("is_door", semantic_name == "portal"))
        ),
        "is_movable_door": False if minimal_gt else bool(observation.get("is_movable_door", False)),
        "joint_type": "none" if minimal_gt else normalize_joint_type(observation.get("joint_type")),
        "joint_range": [0.0, 0.0] if minimal_gt else point_range(observation.get("joint_range")),
        "joint_value": None if minimal_gt else float(observation["joint_value"]) if observation.get("joint_value") is not None else None,
        "joint_infos": [] if minimal_gt else list(observation.get("joint_infos") or []),
        "primary_joint_name": "" if minimal_gt else str(observation.get("primary_joint_name") or ""),
        "orientation": [0.0, 0.0, 0.0, 1.0] if minimal_gt else list(observation.get("orientation") or [0.0, 0.0, 0.0, 1.0]),
        "interaction_approach_axis_xy": interaction_approach_axis_xy,
        "interaction_approach_axis_source": interaction_approach_axis_source,
        "source_object_name": str(
            instance_id
            if minimal_gt
            else observation.get("source_object_name") or observation.get("id") or ""
        ),
        "private_source_object_name": private_instance_id,
        "visible_pixels": visible_pixels,
        "visible_fraction": float(visible_fraction or 0.0),
        "bbox_2d": bbox_2d,
        "segmentation": segmentation,
        "box_3d_frame_id": str(box_3d.get("frame_id") or ""),
        "projected_bbox_2d": [] if minimal_gt else list(observation.get("projected_bbox_2d") or []),
        "consecutive_observations": int(
            0 if minimal_gt else observation.get("consecutive_observations", 0) or 0
        ),
        "camera_name": "" if minimal_gt else str(observation.get("camera_name") or ""),
        # A strict minimal-GT wire record must not smuggle in a caller-owned
        # frame index.  The mapping callback may attach `_capture_step` from
        # its validated envelope as private version metadata before calling
        # this normalizer.
        "frame_index": (
            int(observation.get("_capture_step", 0) or 0)
            if minimal_gt
            else int(observation.get("frame_index", 0) or 0)
        ),
        "episode_id": "" if minimal_gt else str(observation.get("episode_id") or ""),
        "source": "realtime_gt_observation" if minimal_gt else str(observation.get("source") or "detector"),
        "name": (
            instance_id
            if minimal_gt and semantic_name == "portal"
            else str(
                observation.get("name")
                if minimal_gt
                else observation.get("name")
                or observation.get("object_name")
                or semantic_name
                or "object"
            )
        ),
        "asset_id": None if minimal_gt else observation.get("asset_id"),
        "object_id": None if minimal_gt else observation.get("object_id"),
        "viz_aabb_center": point3(viz_aabb_center),
        "viz_aabb_size": point3(viz_aabb_size),
    }


def point_range(values):
    vals = list(values or [])
    if len(vals) < 2:
        vals.extend([0.0] * (2 - len(vals)))
    return [float(vals[0]), float(vals[1])]


def infer_node_type(observation: dict[str, Any]) -> str:
    label = normalize_label(observation.get("semantic_name"))
    if label in PORTAL_LABELS:
        return "portal"
    if label in SUPPORT_LABELS:
        return "support"
    if label in CONTAINER_LABELS:
        return "container"
    return "object"


def default_interaction_payload(node_type: str, observation: dict[str, Any]) -> dict[str, Any]:
    label = normalize_label(observation.get("semantic_name"))
    interaction_mode = "none"
    if node_type == "portal":
        interaction_mode = "open_close"
    elif node_type == "container":
        if label in {"box", "storage_bin"}:
            interaction_mode = "none"
        elif label in {"drawer", "dresser", "chest_of_drawers"}:
            interaction_mode = "slide"
        elif label in OPENABLE_CONTAINER_LABELS:
            interaction_mode = "open_close"

    elif node_type == "support":
        interaction_mode = "place_on"
    state = "unknown"
    is_interactable = interaction_mode != "none"
    # Restricted realtime-GT is only a geometry/visibility contract.  A portal
    # class is not proof of a door leaf, an operable joint, or a closed state.
    # It is nevertheless a valid *interaction hypothesis*: the rule lane must
    # try an unknown portal once and learn ``open/static_open/blocked`` only
    # from executor feedback.  This is an ontology prior, not articulation GT.
    portal_requires_observed_evidence = bool(
        node_type == "portal" and observation.get("minimal_gt_observation")
    )
    if node_type == "portal":
        requires_interaction = bool(
            is_interactable and state not in {"open", "static_open"}
        )
        traversable = (
            None
            if portal_requires_observed_evidence
            else state in {"open", "static_open"}
        )
    else:
        requires_interaction = bool(is_interactable and state in {"closed", "unknown"})
        traversable = True if state in {"open", "ajar", "static_open"} else False if state == "closed" else None
    return {
        "is_interactable": is_interactable,
        "interaction_mode": interaction_mode,
        # This is deliberately an ontology prior, not an articulation fact.
        # In restricted-GT mode the real capability remains explicitly
        # unobserved until the executor returns an action result.
        "capability": "unknown",
        "capability_source": (
            "unobserved" if portal_requires_observed_evidence else "semantic_label_prior"
        ),
        "capability_confidence": (
            0.0
            if portal_requires_observed_evidence
            else float(observation.get("confidence", 0.0) or 0.0)
        ),
        "capability_observed_step": None,
        "capability_evidence": "none",
        "state": state,
        "cost": 1.0,
        "confidence": float(observation.get("confidence", 0.0) or 0.0),
        "state_source": (
            "unobserved"
            if portal_requires_observed_evidence
            else "semantic_graph_default"
        ),
        "state_confidence": (
            0.0
            if portal_requires_observed_evidence
            else float(observation.get("confidence", 0.0) or 0.0)
        ),
        "state_observed_step": None,
        "state_evidence": "none",
        "interaction_cost": 1.0,
        "requires_interaction": requires_interaction,
        "traversable": traversable,
        "expected_effect": "unlock_connectivity" if node_type == "portal" else "reveal_contents" if node_type == "container" else "none",
        "operation_history": [],
        "completed_interaction_groups": [],
        "failed_interaction_groups": [],
    }

def observation_from_detection(detection: dict[str, Any], observation_id: str, source: str = "detector") -> dict[str, Any]:
    world_position = detection.get("world_position") or detection.get("position") or {}
    world_box_center = detection.get("world_box3d_center") or detection.get("aabb_center") or detection.get("box3d_center") or world_position
    size = detection.get("world_box3d_size") or detection.get("aabb_size") or detection.get("box3d_size") or detection.get("size") or {}
    viz_box_center = detection.get("viz_aabb_center") or world_box_center
    viz_box_size = detection.get("viz_aabb_size") or size
    if isinstance(world_position, dict):
        position = [
            world_position.get("x", 0.0),
            world_position.get("y", 0.0),
            world_position.get("z", 0.0),
        ]
    else:
        position = point3(world_position)
    if isinstance(world_box_center, dict):
        aabb_center = [
            world_box_center.get("x", 0.0),
            world_box_center.get("y", 0.0),
            world_box_center.get("z", 0.0),
        ]
    else:
        aabb_center = point3(world_box_center)
    if isinstance(size, dict):
        aabb_size = [
            size.get("x", 0.0),
            size.get("y", 0.0),
            size.get("z", 0.0),
        ]
    else:
        aabb_size = point3(size)
    viz_aabb_center = point3(viz_box_center)
    viz_aabb_size = point3(viz_box_size)
    semantic_name = detection.get("semantic_class") or detection.get("semantic_name") or detection.get("class") or "object"
    return normalize_observation(
        {
            "observation_id": observation_id,
            "instance_id": detection.get("instance_id") or "",
            "semantic_name": semantic_name,
            "category": detection.get("category") or detection.get("semantic_class") or detection.get("class") or semantic_name,
            "candidate_labels": list(detection.get("candidate_labels") or []),
            "label_votes": dict(detection.get("label_votes") or {}),
            "confidence": detection.get("confidence", detection.get("conf", 0.0)),
            "position": position,
            "aabb_center": aabb_center,
            "aabb_size": aabb_size,
            "room_id": detection.get("room_id"),
            "connected_room_ids": detection.get("connected_room_ids") or [],
            "parent": detection.get("parent"),
            "children": list(detection.get("children") or []),
            "is_receptacle": detection.get("is_receptacle", False),
            "is_pickup_candidate": detection.get("is_pickup_candidate", False),
            "is_articulable": detection.get("is_articulable", False),
            "is_door": detection.get("is_door", False),
            "is_movable_door": detection.get("is_movable_door", False),
            "joint_type": detection.get("joint_type"),
            "joint_range": detection.get("joint_range") or [0.0, 0.0],
            "joint_value": detection.get("joint_value"),
            "name": detection.get("name") or detection.get("object_name") or detection.get("instance_id") or semantic_name,
            "asset_id": detection.get("asset_id"),
            "object_id": detection.get("object_id"),
            "source": source,
            "viz_aabb_center": viz_aabb_center,
            "viz_aabb_size": viz_aabb_size,
        }
    )


def room_node_label(room_id_to_name: dict[int, str], room_id: int | None) -> str:
    if room_id is None:
        return "unknown"
    return normalize_label(room_id_to_name.get(int(room_id), f"room_{room_id}"))


def distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
