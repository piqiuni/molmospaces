import math
import time
from statistics import median
from collections import defaultdict

from .geometry_utils import euclidean_2d, grid_index, normalize_label, point_dict, world_to_grid
from .semantic_evidence import _accepted_semantic_evidence


def _detection_yaw(detection):
    """Read the upright OBB yaw emitted by the physical detector."""
    value = detection.get("yaw")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    quaternion = detection.get("world_box3d_orientation") or detection.get("orientation")
    if isinstance(quaternion, (list, tuple)) and len(quaternion) >= 4:
        try:
            x, y, z, w = [float(item) for item in quaternion[:4]]
            return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        except (TypeError, ValueError):
            pass
    return None


def _yaw_quaternion(yaw):
    if yaw is None:
        return [0.0, 0.0, 0.0, 1.0]
    yaw = float(yaw)
    return [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)]


def _axial_yaw_delta(first, second):
    """Return the unsigned yaw delta for an unoriented box axis."""
    delta = float(first) - float(second)
    return abs(0.5 * math.atan2(math.sin(2.0 * delta), math.cos(2.0 * delta)))


def _is_portal_label(label):
    normalized = normalize_label(label)
    return any(token in normalized for token in ("door", "portal", "gate"))


# Cross-label duplicate tracking is intentionally limited to these alias
# families.  Keep the sets at module scope: the duplicate pass runs on every
# detector receipt and rebuilding them inside the pairwise loop adds needless
# allocations when the scene contains many unrelated labels.
_FURNITURE_DUPLICATE_LABELS = frozenset(
    {"sofa", "bed", "couch", "settee", "divan", "bench"}
)
_APPLIANCE_DUPLICATE_LABELS = frozenset(
    {
        "fridge",
        "refrigerator",
        "smart_vending_refrigerator",
        "freezer",
        "locker",
        "safe",
        "water_dispenser",
        "dispenser",
    }
)


class ObjectMapStore:
    STABLE_BOX_MIN_OVERLAP = 0.5
    STABLE_BOX_MIN_SIZE_RATIO = 0.4
    STABLE_BOX_UPDATE_ALPHA = 0.45
    STABLE_BOX_MAX_STEP_RATIO = 1.35
    STABLE_BOX_MIN_ABS_STEP = 0.08

    def __init__(
        self,
        match_distance=0.5,
        stale_after_sec=0.0,
        min_confirmations=2,
        size_match_ratio=0.7,
        stable_history_size=5,
        duplicate_bbox_iou_threshold=0.0,
        duplicate_3d_overlap_threshold=0.15,
        class_min_confirmations=None,
        class_min_top_height_m=None,
        portal_cross_view_match_enabled=False,
        portal_cross_view_normal_distance_m=0.45,
        portal_cross_view_yaw_tolerance_rad=0.35,
        portal_cross_view_min_tangent_overlap_ratio=0.20,
        portal_cross_view_min_vertical_overlap_ratio=0.35,
        persistent_classes=None,
        persistence_requires_m1=True,
        confirmed_geometry_reacquisition=False,
        stable_box_recovery_confirmations=0,
        stable_box_recovery_min_confidence=0.25,
        stable_box_recovery_min_depth_points=64,
        size_aware_xyz_matching=False,
    ):
        self.match_distance = float(match_distance)
        self.size_aware_xyz_matching = bool(size_aware_xyz_matching)
        self.stale_after_sec = float(stale_after_sec)
        # None preserves the historical M1-only lifetime policy. An explicit
        # list selects deployment-specific tracking memory, not action admission.
        self.persistent_classes = (
            None if persistent_classes is None else
            frozenset(normalize_label(label) for label in persistent_classes)
        )
        self.persistence_requires_m1 = bool(persistence_requires_m1)
        self.confirmed_geometry_reacquisition = bool(confirmed_geometry_reacquisition)
        self.stable_box_recovery_confirmations = max(0, int(stable_box_recovery_confirmations))
        self.stable_box_recovery_min_confidence = float(stable_box_recovery_min_confidence)
        self.stable_box_recovery_min_depth_points = max(0, int(stable_box_recovery_min_depth_points))
        self.min_confirmations = max(1, int(min_confirmations))
        self.size_match_ratio = max(0.05, float(size_match_ratio))
        self.stable_history_size = max(1, int(stable_history_size))
        self.duplicate_bbox_iou_threshold = max(0.0, float(duplicate_bbox_iou_threshold))
        self.duplicate_3d_overlap_threshold = max(0.0, float(duplicate_3d_overlap_threshold))
        self.portal_cross_view_match_enabled = bool(portal_cross_view_match_enabled)
        self.portal_cross_view_normal_distance_m = max(
            0.0, float(portal_cross_view_normal_distance_m)
        )
        self.portal_cross_view_yaw_tolerance_rad = max(
            0.0, float(portal_cross_view_yaw_tolerance_rad)
        )
        self.portal_cross_view_min_tangent_overlap_ratio = min(
            1.0, max(0.0, float(portal_cross_view_min_tangent_overlap_ratio))
        )
        self.portal_cross_view_min_vertical_overlap_ratio = min(
            1.0, max(0.0, float(portal_cross_view_min_vertical_overlap_ratio))
        )
        self.class_min_confirmations = {
            normalize_label(label): max(1, int(count))
            for label, count in (class_min_confirmations or {}).items()
            if normalize_label(label)
        }
        self.class_min_top_height_m = {
            normalize_label(label): max(0.0, float(height))
            for label, height in (class_min_top_height_m or {}).items()
            if normalize_label(label)
        }
        self.objects = []
        self.next_id = 1
        self._last_update_stamp = None
        self.m1_canonical_labels = {}
        self.merged_track_aliases = {}
        # Matching indexes are rebuilt at each update-batch boundary and then
        # maintained as tracks move/create.  They are deliberately secondary
        # to the exact matcher below: a missing/invalid index falls back to the
        # historical full insertion-order scan.
        self._match_identity_index = None
        self._match_spatial_index = None
        self._match_spatial_bucket = {}
        self._match_portal_ids = set()
        self._match_index_order = {}
        self._match_object_index = {}
        self._match_index_cell_size = max(abs(self.match_distance), 1e-6)

    def reset(self):
        """Clear observation identity/state while preserving tracker tuning."""
        self.objects = []
        self.next_id = 1
        self._last_update_stamp = None
        self.m1_canonical_labels.clear()
        self.merged_track_aliases.clear()
        self._invalidate_match_indexes()

    def set_m1_canonical_label(self, track_id, label):
        track_id = str(track_id or "")
        label = normalize_label(label)
        if track_id and label:
            self.m1_canonical_labels[track_id] = label
            # Keep the tracker record alive after it has passed the same
            # two-frame + M1 gate used by the semantic graph.  Visibility is
            # still allowed to drop to zero; only identity lifetime changes.
            for obj in self.objects:
                if str(obj.get("track_id") or "") != track_id:
                    continue
                obj["m1_canonical_label"] = label
                obj["m1_retention_label"] = label
                obj["m1_confirmed"] = True
                obj["semantic_name"] = label
                obj["label_votes"] = {label: max(float(obj.get("conf", 0.0)), 0.05)}
                break
            self._invalidate_match_indexes()

    def update_retention_evidence(self, track_id, attributes):
        """Allow an accepted M1 correction to revoke configured track memory."""
        if self.persistent_classes is None:
            return
        evidence = _accepted_semantic_evidence(attributes)
        label = normalize_label(evidence.get("m1_observed_object_name"))
        confidence = float(evidence.get("attribute_confidence", 0.0) or 0.0)
        if (not label or evidence.get("attribute_status") != "ready"
                or not 0.5 <= confidence <= 1.0
                or evidence.get("m1_refrigerator_pending_confirmation", False)):
            return
        for obj in self.objects:
            if str(obj.get("track_id") or "") == str(track_id):
                obj["m1_retention_label"] = label
                obj["m1_confirmed"] = True
                break

    def _required_confirmations(self, label):
        return self.class_min_confirmations.get(
            normalize_label(label), self.min_confirmations
        )

    def _passes_top_height_filter(self, label, detection, center, size):
        """Filter by the highest observed world-Z point.

        The detector's robust/OBB dimensions are useful for tracking, but
        their quantile clipping can lower the apparent top of a tall object.
        Prefer the explicit point-cloud maximum exported by the detector and
        only derive a top from an axis-aligned box as a legacy fallback.
        """
        minimum = self.class_min_top_height_m.get(normalize_label(label))
        if minimum is None:
            return True
        for key in ("world_aabb_max_z", "world_box3d_max_z", "aabb_max_z"):
            value = detection.get(key)
            if value is not None:
                try:
                    top_z = float(value)
                except (TypeError, ValueError):
                    top_z = math.nan
                if math.isfinite(top_z):
                    return top_z >= minimum
        has_center = any(
            detection.get(key) is not None
            for key in ("aabb_center", "world_position", "position", "world_box3d_center", "box3d_center")
        )
        has_size = any(
            detection.get(key) is not None
            for key in ("aabb_size", "world_box3d_size", "box3d_size", "size")
        )
        if not has_center or not has_size:
            return True
        try:
            # Legacy payloads did not carry the point-cloud maximum. Prefer
            # the axis-aligned center/size pair over an oriented-box center;
            # its upper z face is the actual box top by construction.
            legacy_center = self._point_from_detection(
                detection, "aabb_center", "world_position", "position",
                "world_box3d_center", "box3d_center",
            )
            legacy_size = self._point_from_detection(
                detection, "aabb_size", "world_box3d_size", "box3d_size", "size",
            )
            top_z = float(legacy_center["z"]) + 0.5 * abs(float(legacy_size["z"]))
        except (KeyError, TypeError, ValueError):
            return True
        return not math.isfinite(top_z) or top_z >= minimum

    def update(self, detections, stamp, *, geometry_deferred_detections=None):
        now = float(stamp if stamp is not None else time.time())
        if not math.isfinite(now):
            return False
        if (self.objects and self._last_update_stamp is not None
            and now <= self._last_update_stamp):
            return False
        # Clearing objects is the existing episode-reset interface. A reset
        # must also allow the new episode's capture clock to start over.
        self._last_update_stamp = now
        self._rebuild_match_indexes()
        matched_ids = set()
        deferred = list(geometry_deferred_detections or [])
        for obj in self.objects:
            obj["geometry_observed_in_frame"] = False
        for det in detections:
            # A detector can deliberately retain a 2-D-only record for
            # visualization after skipping its expensive RGB-D lift.  It is
            # not an observation for tracking; accepting it here would make
            # the legacy fallbacks create a zero-sized track at the map
            # origin.
            if not isinstance(det, dict):
                continue
            if bool(det.get("geometry_skipped", False)):
                deferred.append(det)
                continue
            detector_label = normalize_label(
                det.get("semantic_class") or det.get("class") or det.get("semantic_name")
            )
            source_instance_id = str(det.get("instance_id") or det.get("track_id") or "")
            canonical = self.m1_canonical_labels.get(source_instance_id, "")
            label = canonical or normalize_label(
                det.get("semantic_class") or det.get("class") or det.get("semantic_name")
            )
            if not label:
                continue
            if canonical:
                det = dict(det)
                det["semantic_class"] = canonical
                det["class"] = canonical
                det["semantic_name"] = canonical
                det["category"] = canonical
                det["m1_canonicalized"] = True
            try:
                pos = self._point_from_detection(
                    det, "world_position", "position", "world_box3d_center", "box3d_center", "aabb_center",
                    required=True,
                )
                confidence = float(det.get("confidence", det.get("conf", 0.0)) or 0.0)
                yaw = _detection_yaw(det)
                size = self._point_from_detection(det, "world_box3d_size", "box3d_size", "size", "aabb_size")
                center = self._point_from_detection(det, "world_box3d_center", "box3d_center", "aabb_center", "world_position", "position")
                viz_center = self._point_from_detection(
                    det, "world_box3d_center", "aabb_center", "box3d_center", "world_position", "position"
                )
                viz_size = self._point_from_detection(
                    det, "world_box3d_size", "aabb_size", "box3d_size", "size"
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
                or (yaw is not None and not math.isfinite(yaw))
                or any(not math.isfinite(float(value)) for point in (pos, size, center, viz_center, viz_size) for value in point.values())
                or any(float(value) < 0.0 for point in (size, viz_size) for value in point.values())):
                continue
            instance_id = source_instance_id
            match = self._find_match(label, pos, size, instance_id, yaw=yaw)
            if match is not None and int(match["object_id"]) in matched_ids:
                # Several overlapping detector boxes in one image are not
                # independent temporal confirmations of this physical track.
                if not self.size_aware_xyz_matching or self._bbox_iou_2d(
                    det.get("bbox_2d") or det.get("bbox"),
                    match.get("visual_bbox_2d") or match.get("bbox_2d"),
                ) >= 0.85:
                    continue
                # A separate same-frame box cannot consume an already used
                # track. Search remaining tracks before creating a new one.
                match = self._find_match(label, pos, size, instance_id, yaw=yaw,
                                         excluded_ids=matched_ids)
            if match is not None and match.get("m1_canonical_label"):
                # Raw detector streams need not echo our generated track ID.
                # Once geometry matches, retain the accepted M1 identity.
                label = str(match["m1_canonical_label"])
            # Geometry is an admission/confirmation gate only. Once the track
            # is confirmed, close-range partial boxes keep updating it.
            if (
                (match is None or not match.get("is_confirmed"))
                and not self._passes_top_height_filter(label, det, center, size)
            ):
                continue
            if match is None:
                match = {
                    "object_id": self.next_id,
                    "track_id": f"track_{self.next_id:04d}",
                    "semantic_name": label,
                    "label_votes": {},
                    "conf": confidence,
                    "coord": [pos["x"], pos["y"], pos["z"]],
                    "aabb_center": [center["x"], center["y"], center["z"]],
                    "aabb_size": [size["x"], size["y"], size["z"]],
                    "viz_aabb_center": [viz_center["x"], viz_center["y"], viz_center["z"]],
                    "viz_aabb_size": [viz_size["x"], viz_size["y"], viz_size["z"]],
                    "yaw": yaw,
                    "yaw_history": [],
                    "coord_history": [],
                    "observation_count": 0,
                    "hit_streak": 0,
                    "miss_streak": 0,
                    "is_confirmed": False,
                    "instance_id": instance_id,
                    "last_seen": now,
                    "bbox_2d": [],
                    "visible_pixels": 0,
                    "max_visible_pixels": 0,
                    "visible_fraction": 0.0,
                    "max_visible_fraction": 0.0,
                    "max_consecutive_observations": 0,
                    "latest_segmentation": None,
                    "capture_step": None,
                }
                self.next_id += 1
                self.objects.append(match)

            self._append_history(
                match,
                "coord_history",
                [pos["x"], pos["y"], pos["z"]],
            )
            match["coord"] = self._history_median(match.get("coord_history"))
            if match["observation_count"] <= 0 or self._should_update_stable_box(match, center, size):
                match.pop("box_recovery_window", None)
                blended_center, blended_size = self._blend_stable_box(
                    match.get("aabb_center", [center["x"], center["y"], center["z"]]),
                    match.get("aabb_size", [size["x"], size["y"], size["z"]]),
                    [center["x"], center["y"], center["z"]],
                    [size["x"], size["y"], size["z"]],
                    is_first=match["observation_count"] <= 0,
                )
                match["aabb_center"] = blended_center
                match["aabb_size"] = blended_size
            else:
                self._recover_stable_box(match, center, size, det, now)
            match["viz_aabb_center"] = [viz_center["x"], viz_center["y"], viz_center["z"]]
            match["viz_aabb_size"] = [viz_size["x"], viz_size["y"], viz_size["z"]]
            if yaw is not None:
                self._append_scalar_history(match, "yaw_history", yaw)
                match["yaw"] = self._history_axial_mean(match.get("yaw_history"))
            match["conf"] = min(1.0, max(float(match["conf"]), confidence) + 0.05 * confidence)
            label_votes = dict(match.get("label_votes") or {})
            label_votes[label] = float(label_votes.get(label, 0.0)) + max(confidence, 0.05)
            match["label_votes"] = label_votes
            match["semantic_name"] = max(
                sorted(label_votes.keys()),
                key=lambda key: (float(label_votes[key]), key == match.get("semantic_name")),
            )
            match["observation_count"] += 1
            match["hit_streak"] = int(match.get("hit_streak", 0)) + 1
            match["miss_streak"] = 0
            bbox_2d = list(det.get("bbox_2d") or det.get("bbox") or [])
            mask = det.get("mask")
            mask_pixels = 0
            if isinstance(mask, dict):
                rows = mask.get("rows") or []
                cols = mask.get("cols") or []
                if len(rows) == len(cols):
                    mask_pixels = len(rows)
            bbox_area = 0.0
            if len(bbox_2d) >= 4:
                bbox_area = max(0.0, float(bbox_2d[2]) - float(bbox_2d[0])) * max(
                    0.0, float(bbox_2d[3]) - float(bbox_2d[1])
                )
            has_mask_evidence = any(
                det.get(key) is not None
                for key in ("segmentation", "mask", "mask_rle", "segmentation_rle")
            )
            has_visible_pixel_count = "visible_pixels" in det or "mask_area" in det
            if has_mask_evidence or has_visible_pixel_count:
                visible_pixels = int(
                    # Physical mask coordinates are capped debug samples;
                    # mask_area retains the full segmentation pixel count.
                    det.get("visible_pixels", det.get("mask_area", mask_pixels)) or 0
                )
            else:
                # The physical YOLO detector publishes boxes but no masks. In
                # that wire format the visible image support is the box itself;
                # keeping it at zero incorrectly filters every door before M1.
                visible_pixels = int(bbox_area)
            visible_fraction = float(
                det.get(
                    "visible_fraction",
                    min(1.0, visible_pixels / bbox_area) if bbox_area > 0.0 else 0.0,
                )
                or 0.0
            )
            match["bbox_2d"] = bbox_2d
            match["visual_bbox_2d"] = bbox_2d
            segmentation = (
                det.get("segmentation")
                or det.get("mask")
                or det.get("mask_rle")
                or det.get("segmentation_rle")
            )
            if segmentation is not None:
                match["latest_segmentation"] = segmentation
            capture_step = det.get("capture_step", det.get("capture_seq"))
            if capture_step is not None:
                try:
                    match["capture_step"] = int(capture_step)
                except (TypeError, ValueError):
                    pass
            match["visible_pixels"] = visible_pixels
            match["max_visible_pixels"] = max(
                int(match.get("max_visible_pixels", 0) or 0), visible_pixels
            )
            match["visible_fraction"] = visible_fraction
            match["max_visible_fraction"] = max(
                float(match.get("max_visible_fraction", 0.0) or 0.0), visible_fraction
            )
            match["max_consecutive_observations"] = max(
                int(match.get("max_consecutive_observations", 0) or 0),
                int(match["hit_streak"]),
            )
            # Confirmation is a temporal gate, not an accumulated lifetime
            # count. Sparse one-frame detections must never become stable just
            # because the same spatial track is revisited many times.
            match["is_confirmed"] = bool(
                int(match["hit_streak"])
                >= self._required_confirmations(match["semantic_name"])
                or self._has_configured_tracking_memory(match)
                or (
                    self.confirmed_geometry_reacquisition
                    and self._has_temporal_confirmation(match)
                )
            )
            match["last_seen"] = now
            match["last_visual_seen"] = now
            match["last_detector_label"] = detector_label
            match["geometry_observed_in_frame"] = True
            matched_ids.add(int(match["object_id"]))
            self._index_match_object(match)

        deferred_ids = self._associate_deferred_detections(deferred, now, matched_ids)
        for obj in self.objects:
            if int(obj["object_id"]) in matched_ids or int(obj["object_id"]) in deferred_ids:
                continue
            obj["hit_streak"] = 0
            obj["miss_streak"] = int(obj.get("miss_streak", 0)) + 1
            obj.pop("box_recovery_window", None)

        self._merge_duplicate_tracks()
        self._purge_stale(now)
        return True

    def _has_temporal_confirmation(self, obj, required=None):
        required = max(2, int(required if required is not None else
                              self._required_confirmations(obj.get("semantic_name"))))
        return bool(
            int(obj.get("observation_count", 0)) >= required
            and int(obj.get("max_consecutive_observations", 0)) >= required
        )

    def _associate_deferred_detections(self, deferred, now, matched_ids):
        """A budget omission can hold a visual streak, never create 3-D evidence."""
        if not self.confirmed_geometry_reacquisition or not deferred:
            return set()
        by_label = defaultdict(list)
        for obj in self.objects:
            if int(obj["object_id"]) in matched_ids:
                continue
            if now - float(obj.get("last_visual_seen", obj.get("last_seen", now))) > 0.75:
                continue
            label = obj.get("last_detector_label") or obj.get("semantic_name")
            by_label[normalize_label(label)].append(obj)
        associated = set()
        for det in deferred:
            if not isinstance(det, dict) or not det.get("geometry_skipped"):
                continue
            if det.get("geometry_skip_reason") not in {
                "max_geometry_instances", "geometry_budget_ms", "geometry_workers_busy",
            }:
                continue
            label = normalize_label(det.get("semantic_class") or det.get("semantic_name"))
            bbox = det.get("bbox_2d") or det.get("bbox") or []
            candidates = []
            for obj in by_label.get(label, []):
                if int(obj["object_id"]) in associated:
                    continue
                try:
                    iou = self._bbox_iou_2d(bbox, obj.get("visual_bbox_2d") or obj.get("bbox_2d"))
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(iou) and iou >= 0.6:
                    candidates.append((iou, obj))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.15):
                continue
            obj = candidates[0][1]
            obj["last_visual_seen"] = now
            obj["visual_bbox_2d"] = list(bbox)
            associated.add(int(obj["object_id"]))
        return associated

    @staticmethod
    def _bbox_iou_2d(a, b):
        if len(a or []) < 4 or len(b or []) < 4:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
        bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return intersection / max(area_a + area_b - intersection, 1e-6)

    def _merge_duplicate_tracks(self):
        """Collapse split tracks when current 2D/3D boxes identify one object.

        A detector can legitimately alternate between neighbouring open-vocabulary
        labels (notably ``sofa`` and ``bed``).  Keep those label votes on one
        spatial track instead of creating a new object for every label variant.
        """
        if self.duplicate_bbox_iou_threshold <= 0.0 or len(self.objects) < 2:
            return
        removed = set()
        # Eligibility is an equivalence relation: exact labels may merge with
        # the same label, while cross-label merges are limited to one of the
        # two explicit alias families.  Partitioning first preserves the
        # original pairwise semantics but avoids scanning every unrelated
        # label pair (the common scene has many one-off open-vocabulary
        # labels).  Each group is sorted with the same keeper ordering as the
        # former global pass, so ties and evidence dominance are unchanged.
        groups = {}
        for obj in self.objects:
            label = str(obj.get("semantic_name") or "")
            if label in _FURNITURE_DUPLICATE_LABELS:
                group_key = ("family", "furniture")
            elif label in _APPLIANCE_DUPLICATE_LABELS:
                group_key = ("family", "appliance")
            else:
                group_key = ("label", label)
            groups.setdefault(group_key, []).append(obj)
        ordered_groups = [
            sorted(
                group,
                key=lambda obj: (
                    -int(obj.get("observation_count", 0)),
                    int(obj.get("object_id", 0)),
                ),
            )
            for group in groups.values()
            if len(group) >= 2
        ]
        if not ordered_groups:
            return
        for ordered in ordered_groups:
            for index, keeper in enumerate(ordered):
                if int(keeper.get("object_id", -1)) in removed:
                    continue
                for duplicate in ordered[index + 1 :]:
                    duplicate_id = int(duplicate.get("object_id", -1))
                    if duplicate_id in removed:
                        continue
                    same_label = keeper.get("semantic_name") == duplicate.get("semantic_name")
                    keeper_label = str(keeper.get("semantic_name") or "")
                    duplicate_label = str(duplicate.get("semantic_name") or "")
                    # Different labels can only merge inside the explicitly
                    # supported alias families.  Do this cheap check before IoU,
                    # distance, and 3-D overlap calculations; the conditions
                    # below remain unchanged for every eligible pair.
                    if not same_label and not (
                        keeper_label in _FURNITURE_DUPLICATE_LABELS
                        and duplicate_label in _FURNITURE_DUPLICATE_LABELS
                    ) and not (
                        keeper_label in _APPLIANCE_DUPLICATE_LABELS
                        and duplicate_label in _APPLIANCE_DUPLICATE_LABELS
                    ):
                        continue
                    bbox_iou = self._bbox_iou_2d(
                        keeper.get("bbox_2d"), duplicate.get("bbox_2d")
                    )
                    # Cross-label merges are deliberately restricted to the
                    # furniture family where sofa/bed/couch are common aliases.
                    # Require stronger 2-D overlap than ordinary same-label NMS
                    # so adjacent furniture is not collapsed accidentally.
                    # Open-vocabulary detector labels can split one tall appliance
                    # into ``locker``/``safe`` (and M1 may later call both
                    # refrigerator).  A near-identical image box plus overlapping
                    # 3-D body is stronger duplicate evidence than the label in
                    # this case; keep adjacent appliances separate by requiring the
                    # configured high IoU and the existing 3-D overlap gate.
                    cross_label = (
                        not same_label
                        and keeper_label in _FURNITURE_DUPLICATE_LABELS
                        and duplicate_label in _FURNITURE_DUPLICATE_LABELS
                        and bbox_iou >= max(self.duplicate_bbox_iou_threshold, 0.70)
                    )
                    cross_label_appliance = (
                        not same_label
                        and keeper_label in _APPLIANCE_DUPLICATE_LABELS
                        and duplicate_label in _APPLIANCE_DUPLICATE_LABELS
                        and bbox_iou >= max(self.duplicate_bbox_iou_threshold, 0.85)
                    )
                    if cross_label_appliance:
                        cross_label = True
                    if not same_label and not cross_label:
                        continue
                    if same_label and bbox_iou < self.duplicate_bbox_iou_threshold:
                        continue
                    if cross_label and not cross_label_appliance and bbox_iou < 0.70:
                        continue
                    if cross_label_appliance and bbox_iou < max(
                        self.duplicate_bbox_iou_threshold, 0.85
                    ):
                        continue
                    center_a = keeper.get("aabb_center", keeper.get("coord", [0.0, 0.0, 0.0]))
                    center_b = duplicate.get("aabb_center", duplicate.get("coord", [0.0, 0.0, 0.0]))
                    center_distance = math.dist([float(v) for v in center_a], [float(v) for v in center_b])
                    # Depth can jump between foreground/background surfaces while
                    # the 2-D mask remains the same object.  For near-identical
                    # image boxes, tolerate that depth disagreement and use a
                    # wider spatial gate; ordinary boxes retain the strict gate.
                    if center_distance >= self.match_distance and bbox_iou < 0.90:
                        continue
                    overlap = self._aabb_overlap_ratio(
                        center_a,
                        keeper.get("aabb_size", [0.0, 0.0, 0.0]),
                        center_b,
                        duplicate.get("aabb_size", [0.0, 0.0, 0.0]),
                    )
                    if overlap < self.duplicate_3d_overlap_threshold and bbox_iou < 0.90:
                        continue
                    keeper["conf"] = max(float(keeper.get("conf", 0.0)), float(duplicate.get("conf", 0.0)))
                    keeper["observation_count"] = max(
                        int(keeper.get("observation_count", 0)),
                        int(duplicate.get("observation_count", 0)),
                    )
                    keeper["max_visible_pixels"] = max(
                        int(keeper.get("max_visible_pixels", 0)),
                        int(duplicate.get("max_visible_pixels", 0)),
                    )
                    keeper["max_visible_fraction"] = max(
                        float(keeper.get("max_visible_fraction", 0.0)),
                        float(duplicate.get("max_visible_fraction", 0.0)),
                    )
                    keeper["max_consecutive_observations"] = max(
                        int(keeper.get("max_consecutive_observations", 0)),
                        int(duplicate.get("max_consecutive_observations", 0)),
                    )
                    keeper["last_seen"] = max(float(keeper.get("last_seen", 0.0)), float(duplicate.get("last_seen", 0.0)))
                    keeper["is_confirmed"] = bool(keeper.get("is_confirmed") or duplicate.get("is_confirmed"))
                    votes = dict(keeper.get("label_votes") or {})
                    for label, score in (duplicate.get("label_votes") or {}).items():
                        # Sum evidence from both tracks so candidate_labels exposes
                        # all plausible attributes while semantic_name remains the
                        # highest-vote (dominant) interpretation.
                        votes[label] = float(votes.get(label, 0.0)) + float(score)
                    keeper["label_votes"] = votes
                    keeper["semantic_name"] = max(
                        sorted(votes.keys()),
                        key=lambda key: (float(votes[key]), key == keeper.get("semantic_name")),
                    )
                    removed.add(duplicate_id)
                    old_track = str(duplicate.get("track_id") or "")
                    new_track = str(keeper.get("track_id") or "")
                    if old_track and new_track:
                        self.merged_track_aliases[old_track] = new_track
                        for alias, target in list(self.merged_track_aliases.items()):
                            if target == old_track:
                                self.merged_track_aliases[alias] = new_track
        if removed:
            self.objects = [obj for obj in self.objects if int(obj.get("object_id", -1)) not in removed]
            self._invalidate_match_indexes()

    def as_obj_map(self):
        return [
            {
                "semantic_name": obj["semantic_name"],
                "candidate_labels": self._candidate_labels(obj),
                "label_votes": {str(k): float(v) for k, v in (obj.get("label_votes") or {}).items()},
                "conf": float(obj["conf"]),
                "coord": [float(v) for v in obj["coord"]],
                "object_id": int(obj["object_id"]),
                "track_id": str(obj["track_id"]),
                "observation_count": int(obj["observation_count"]),
                "aabb_center": [float(v) for v in obj["aabb_center"]],
                "aabb_size": [float(v) for v in obj["aabb_size"]],
                "yaw": obj.get("yaw"),
                "world_box3d_orientation": _yaw_quaternion(obj.get("yaw")),
            }
            for obj in self.objects
            if obj.get("is_confirmed")
        ]

    def as_tracked_detections(
        self,
        min_observations=None,
        confirmed_only=True,
        currently_observed_only=False,
        ignore_class_confirmations=False,
    ):
        """Export tracker records that satisfy the requested temporal gate.

        The physical detector uses stricter per-class confirmation counts for
        interaction candidates (doors/refrigerators are normally required to
        persist for several frames).  An opened refrigerator is a special
        case: newly exposed contents can be visible for only one or two
        frames, so the semantic graph may request a *graph-only* tentative
        stream with ``ignore_class_confirmations=True``.  This option changes
        only the admission threshold used for this export; it never mutates a
        track's ``is_confirmed`` state and therefore cannot weaken the M1
        tracked-detection topic.
        """
        if min_observations is None:
            min_observations = self.min_confirmations if confirmed_only else 1
        min_observations = max(1, int(min_observations))
        detections = []
        for obj in self.objects:
            required = (
                min_observations
                if bool(ignore_class_confirmations)
                else max(
                    min_observations,
                    self._required_confirmations(obj.get("semantic_name")),
                )
            )
            if confirmed_only and not obj.get("is_confirmed"):
                continue
            if currently_observed_only:
                if not obj.get("geometry_observed_in_frame", True):
                    continue
                if int(obj.get("miss_streak", 0) or 0) != 0:
                    continue
                reacquired = (
                    self.confirmed_geometry_reacquisition
                    and self._has_temporal_confirmation(obj, required=required)
                )
                if int(obj.get("hit_streak", 0) or 0) < required and not reacquired:
                    continue
            elif int(obj.get("observation_count", 0)) < min_observations:
                continue
            detections.append(
                {
                    "semantic_class": obj["semantic_name"],
                    "candidate_labels": self._candidate_labels(obj),
                    "label_votes": {str(k): float(v) for k, v in (obj.get("label_votes") or {}).items()},
                    "confidence": float(obj["conf"]),
                    # Downstream inference and graph consumers must use the
                    # map-owned identity, never a detector's per-frame ID.
                    "instance_id": str(obj["track_id"]),
                    "source_instance_id": str(obj.get("instance_id") or ""),
                    "object_id": int(obj["object_id"]),
                    "track_id": str(obj["track_id"]),
                    "world_position": point_dict(obj["coord"][0], obj["coord"][1], obj["coord"][2]),
                    "world_box3d_center": point_dict(
                        obj["aabb_center"][0], obj["aabb_center"][1], obj["aabb_center"][2]
                    ),
                    "world_box3d_size": point_dict(
                        obj["aabb_size"][0], obj["aabb_size"][1], obj["aabb_size"][2]
                    ),
                    "yaw": obj.get("yaw"),
                    "world_box3d_orientation": _yaw_quaternion(obj.get("yaw")),
                    "observation_count": int(obj["observation_count"]),
                    "bbox_2d": list(obj.get("bbox_2d") or []),
                    "segmentation": obj.get("latest_segmentation"),
                    "capture_step": obj.get("capture_step"),
                    "visible_pixels": int(obj.get("visible_pixels", 0) or 0),
                    "max_visible_pixels": int(obj.get("max_visible_pixels", 0) or 0),
                    "visible_fraction": float(obj.get("visible_fraction", 0.0) or 0.0),
                    "max_visible_fraction": float(obj.get("max_visible_fraction", 0.0) or 0.0),
                    "consecutive_observations": int(obj.get("hit_streak", 0) or 0),
                    "max_consecutive_observations": int(
                        obj.get("max_consecutive_observations", 0) or 0
                    ),
                    "required_consecutive_observations": required,
                    "tracking_confirmed": bool(obj.get("is_confirmed")),
                    "source": "tracked_object_store",
                    "viz_aabb_center": point_dict(
                        obj["aabb_center"][0], obj["aabb_center"][1], obj["aabb_center"][2]
                    ),
                    "viz_aabb_size": point_dict(
                        obj["aabb_size"][0], obj["aabb_size"][1], obj["aabb_size"][2]
                    ),
                    "latest_box3d_center": point_dict(
                        obj["viz_aabb_center"][0], obj["viz_aabb_center"][1], obj["viz_aabb_center"][2]
                    ),
                    "latest_box3d_size": point_dict(
                        obj["viz_aabb_size"][0], obj["viz_aabb_size"][1], obj["viz_aabb_size"][2]
                    ),
                }
            )
        return detections

    def _append_history(self, obj, key, value):
        history = list(obj.get(key) or [])
        history.append([float(value[0]), float(value[1]), float(value[2])])
        if len(history) > self.stable_history_size:
            history = history[-self.stable_history_size :]
        obj[key] = history

    def _append_scalar_history(self, obj, key, value):
        history = list(obj.get(key) or [])
        history.append(float(value))
        if len(history) > self.stable_history_size:
            history = history[-self.stable_history_size :]
        obj[key] = history

    @staticmethod
    def _history_axial_mean(history):
        history = [float(value) for value in (history or [])]
        if not history:
            return None
        sine = sum(math.sin(2.0 * value) for value in history)
        cosine = sum(math.cos(2.0 * value) for value in history)
        return 0.5 * math.atan2(sine, cosine)

    def _history_median(self, history):
        history = list(history or [])
        if not history:
            return [0.0, 0.0, 0.0]
        dims = list(zip(*history))
        return [float(median(axis_values)) for axis_values in dims]

    def _blend_stable_box(self, old_center, old_size, new_center, new_size, is_first=False):
        if is_first:
            return [float(v) for v in new_center], [float(v) for v in new_size]
        alpha = self.STABLE_BOX_UPDATE_ALPHA
        blended_center = [
            float(old_center[axis]) * (1.0 - alpha) + float(new_center[axis]) * alpha
            for axis in range(3)
        ]
        min_step_ratio = 1.0 / self.STABLE_BOX_MAX_STEP_RATIO
        blended_size = []
        for axis in range(3):
            old_axis = max(float(old_size[axis]), 0.02)
            new_axis = max(float(new_size[axis]), 0.02)
            target = old_axis * (1.0 - alpha) + new_axis * alpha
            lower = max(0.02, min(old_axis * min_step_ratio, old_axis - self.STABLE_BOX_MIN_ABS_STEP))
            upper = max(old_axis * self.STABLE_BOX_MAX_STEP_RATIO, old_axis + self.STABLE_BOX_MIN_ABS_STEP)
            blended_size.append(min(max(target, lower), upper))
        return blended_center, blended_size

    def _invalidate_match_indexes(self):
        """Invalidate secondary match indexes after out-of-band list changes."""

        self._match_identity_index = None
        self._match_spatial_index = None
        self._match_spatial_bucket = {}
        self._match_portal_ids = set()
        self._match_index_order = {}
        self._match_object_index = {}

    def _match_spatial_key(self, coord):
        try:
            x = float(coord[0])
            y = float(coord[1])
            cell_size = float(self._match_index_cell_size)
        except (IndexError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x, y, cell_size)):
            return None
        return (math.floor(x / cell_size), math.floor(y / cell_size))

    def _rebuild_match_indexes(self):
        identity_index = {}
        spatial_index = defaultdict(list)
        spatial_bucket = {}
        portal_ids = set()
        index_order = {}
        object_index = {}
        for order, obj in enumerate(self.objects):
            object_id = int(obj.get("object_id", -1))
            if object_id < 0:
                continue
            object_index[object_id] = obj
            index_order[object_id] = order
            instance_id = str(obj.get("instance_id") or "")
            if instance_id:
                # Preserve the first object, matching the historical scan on
                # a detector-ID collision.
                identity_index.setdefault(instance_id, object_id)
            key = self._match_spatial_key(obj.get("coord") or [])
            if key is not None:
                spatial_index[key].append(object_id)
                spatial_bucket[object_id] = key
            if _is_portal_label(obj.get("semantic_name")):
                portal_ids.add(object_id)
        self._match_identity_index = identity_index
        self._match_spatial_index = dict(spatial_index)
        self._match_spatial_bucket = spatial_bucket
        self._match_portal_ids = portal_ids
        self._match_index_order = index_order
        self._match_object_index = object_index

    def _index_match_object(self, obj):
        """Insert/reposition one track in the active batch indexes."""

        if self._match_spatial_index is None or obj is None:
            return
        object_id = int(obj.get("object_id", -1))
        if object_id < 0:
            return
        self._match_object_index[object_id] = obj
        self._match_index_order.setdefault(object_id, len(self._match_index_order))
        instance_id = str(obj.get("instance_id") or "")
        if instance_id and self._match_identity_index is not None:
            self._match_identity_index.setdefault(instance_id, object_id)

        old_key = self._match_spatial_bucket.get(object_id)
        new_key = self._match_spatial_key(obj.get("coord") or [])
        if old_key != new_key:
            if old_key is not None:
                old_bucket = self._match_spatial_index.get(old_key, [])
                try:
                    old_bucket.remove(object_id)
                except ValueError:
                    pass
                if not old_bucket:
                    self._match_spatial_index.pop(old_key, None)
                self._match_spatial_bucket.pop(object_id, None)
            if new_key is not None:
                self._match_spatial_index.setdefault(new_key, []).append(object_id)
                self._match_spatial_bucket[object_id] = new_key
        if _is_portal_label(obj.get("semantic_name")):
            self._match_portal_ids.add(object_id)
        else:
            self._match_portal_ids.discard(object_id)

    def _indexed_match_candidates(self, label, pos):
        """Return nearby/all-portal candidates in historical insertion order."""

        if self._match_spatial_index is None:
            return None
        candidate_ids = set()
        key = self._match_spatial_key([pos.get("x"), pos.get("y")])
        if key is not None and self.match_distance > 0.0:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidate_ids.update(
                        self._match_spatial_index.get(
                            (key[0] + dx, key[1] + dy), ()
                        )
                    )
        if (
            self.portal_cross_view_match_enabled
            and _is_portal_label(label)
        ):
            candidate_ids.update(self._match_portal_ids)
        if not candidate_ids:
            return []
        ordered_ids = sorted(
            candidate_ids,
            key=lambda object_id: self._match_index_order.get(object_id, 1 << 60),
        )
        return [
            self._match_object_index[object_id]
            for object_id in ordered_ids
            if object_id in self._match_object_index
        ]

    def _find_match(self, label, pos, size, instance_id, yaw=None, excluded_ids=()):
        best = None
        best_score = math.inf
        identity = str(instance_id or "")
        if identity and self._match_identity_index is not None:
            object_id = self._match_identity_index.get(identity)
            if object_id is not None and object_id not in excluded_ids:
                obj = self._match_object_index.get(object_id)
                if obj is not None:
                    return obj
        candidates = self._indexed_match_candidates(label, pos)
        if candidates is None:
            candidates = self.objects
        for obj in candidates:
            if int(obj["object_id"]) in excluded_ids:
                continue
            if identity and obj.get("instance_id") == identity:
                return obj
            obj_pos = point_dict(obj["coord"][0], obj["coord"][1], obj["coord"][2])
            dist = euclidean_2d(pos, obj_pos)
            portal_cross_view = self._portal_cross_view_match(obj, label, pos, size, yaw)
            if dist >= self.match_distance and not portal_cross_view:
                continue
            xyz_score = None
            if self.size_aware_xyz_matching and not portal_cross_view:
                old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
                # Small objects need small gates; the old absolute radius
                # remains an upper bound. A 4 cm floor tolerates depth noise.
                gates = [min(self.match_distance, max(0.04, 0.5 * max(
                    float(old_size[i]), float(size[axis]))))
                    for i, axis in enumerate(("x", "y", "z"))]
                deltas = [abs(float(pos[axis]) - float(obj_pos[axis]))
                          for axis in ("x", "y", "z")]
                if any(gate <= 0 or delta > gate for delta, gate in zip(deltas, gates)):
                    continue
                xyz_score = sum((delta / gate) ** 2 for delta, gate in zip(deltas, gates))
            if obj["semantic_name"] != label:
                if not self._should_merge_cross_label(obj, pos, size):
                    continue
            elif not portal_cross_view and not self._size_compatible(obj.get("aabb_size", [0.0, 0.0, 0.0]), [size["x"], size["y"], size["z"]]):
                old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
                old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
                overlap = self._aabb_overlap_ratio(
                    old_center,
                    old_size,
                    [pos["x"], pos["y"], pos["z"]],
                    [size["x"], size["y"], size["z"]],
                )
                if overlap < 0.15:
                    continue
            score = dist if not portal_cross_view else self._portal_cross_view_score(
                obj, pos, yaw
            )
            if xyz_score is not None:
                score = xyz_score
            if score < best_score:
                best = obj
                best_score = score
        return best

    def _portal_cross_view_match(self, obj, label, pos, size, yaw):
        if not self.portal_cross_view_match_enabled:
            return False
        if obj.get("semantic_name") != label or not _is_portal_label(label):
            return False
        old_yaw = obj.get("yaw")
        if old_yaw is None or yaw is None:
            return False
        if _axial_yaw_delta(old_yaw, yaw) > self.portal_cross_view_yaw_tolerance_rad:
            return False

        old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
        new_center = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        new_size = [float(size["x"]), float(size["y"]), float(size["z"])]
        tangent_x, tangent_y = math.cos(float(old_yaw)), math.sin(float(old_yaw))
        normal_x, normal_y = -tangent_y, tangent_x
        delta_x = new_center[0] - float(old_center[0])
        delta_y = new_center[1] - float(old_center[1])
        normal_distance = abs(delta_x * normal_x + delta_y * normal_y)
        if normal_distance > self.portal_cross_view_normal_distance_m:
            return False

        tangent_distance = abs(delta_x * tangent_x + delta_y * tangent_y)
        old_width = max(abs(float(old_size[0])), abs(float(old_size[1])), 1e-3)
        new_width = max(abs(new_size[0]), abs(new_size[1]), 1e-3)
        tangent_overlap = max(0.0, 0.5 * (old_width + new_width) - tangent_distance)
        tangent_overlap_ratio = tangent_overlap / max(min(old_width, new_width), 1e-3)
        if tangent_overlap_ratio < self.portal_cross_view_min_tangent_overlap_ratio:
            return False

        old_height = max(abs(float(old_size[2])), 1e-3)
        new_height = max(abs(new_size[2]), 1e-3)
        vertical_distance = abs(new_center[2] - float(old_center[2]))
        vertical_overlap = max(0.0, 0.5 * (old_height + new_height) - vertical_distance)
        vertical_overlap_ratio = vertical_overlap / max(min(old_height, new_height), 1e-3)
        return vertical_overlap_ratio >= self.portal_cross_view_min_vertical_overlap_ratio

    @staticmethod
    def _portal_cross_view_score(obj, pos, yaw):
        old_yaw = float(obj.get("yaw", 0.0) or 0.0)
        old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        delta_x = float(pos["x"]) - float(old_center[0])
        delta_y = float(pos["y"]) - float(old_center[1])
        normal_distance = abs(-math.sin(old_yaw) * delta_x + math.cos(old_yaw) * delta_y)
        return normal_distance + 0.25 * _axial_yaw_delta(old_yaw, yaw)

    def _point_from_detection(self, det, *keys, required=False):
        for key in keys:
            value = det.get(key)
            if value is None:
                continue
            if isinstance(value, dict):
                if "x" not in value or "y" not in value:
                    raise ValueError(f"Incomplete geometry field: {key}")
                return point_dict(value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0))
            # Physical YOLOE transport uses compact JSON arrays for 3-D
            # centers/sizes, while older ROS detections use point dicts.
            # Accept both so graph/map visualization preserves the measured
            # box dimensions instead of falling back to a zero-size marker.
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return point_dict(value[0], value[1], value[2])
            # A corrupt supplied measurement must not fall through to a
            # default zero vector, even when another alias happens to exist.
            raise ValueError(f"Invalid geometry field: {key}")
        if required:
            raise ValueError("Missing measured object position")
        return point_dict()

    def _size_compatible(self, old_size, new_size):
        return self._size_ratio(old_size, new_size) >= self.size_match_ratio

    def _size_ratio(self, old_size, new_size):
        old_norm = max(sum(abs(float(v)) for v in old_size), 1e-3)
        new_norm = max(sum(abs(float(v)) for v in new_size), 1e-3)
        return min(old_norm, new_norm) / max(old_norm, new_norm)

    def _should_update_stable_box(self, obj, center, size):
        old_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        old_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
        new_center = [center["x"], center["y"], center["z"]]
        new_size = [size["x"], size["y"], size["z"]]
        if self._size_compatible(old_size, new_size):
            return True
        overlap = self._aabb_overlap_ratio(old_center, old_size, new_center, new_size)
        size_ratio = self._size_ratio(old_size, new_size)
        return overlap >= self.STABLE_BOX_MIN_OVERLAP and size_ratio >= self.STABLE_BOX_MIN_SIZE_RATIO

    def _recover_stable_box(self, obj, center, size, detection, stamp):
        """Replace a bad old box only after a bounded, agreeing RGB-D window."""
        required = self.stable_box_recovery_confirmations
        if required < 2:
            return
        try:
            quality_ok = (
                float(detection.get("confidence", detection.get("conf", 0.0)))
                >= self.stable_box_recovery_min_confidence
                and int(detection.get("depth_valid_points", 0))
                >= self.stable_box_recovery_min_depth_points
            )
        except (TypeError, ValueError, OverflowError):
            quality_ok = False
        if not quality_ok:
            obj.pop("box_recovery_window", None)
            return
        sample = {
            "stamp": stamp,
            "center": [float(center[axis]) for axis in ("x", "y", "z")],
            "size": [float(size[axis]) for axis in ("x", "y", "z")],
        }
        window = list(obj.get("box_recovery_window") or [])
        if window and (
            stamp - window[-1]["stamp"] > 1.0
            or any(
                math.dist(sample["center"], previous["center"]) > 0.2
                or any(
                    min(a, b) / max(a, b, 0.01) < 0.75
                    for a, b in zip(sample["size"], previous["size"])
                )
                for previous in window
            )
        ):
            window = []
        window.append(sample)
        obj["box_recovery_window"] = window[-required:]
        if len(window) < required:
            return
        obj["aabb_center"] = self._history_median([item["center"] for item in window])
        obj["aabb_size"] = self._history_median([item["size"] for item in window])
        obj["box_recovery_count"] = int(obj.get("box_recovery_count", 0)) + 1
        obj.pop("box_recovery_window", None)

    def _candidate_labels(self, obj):
        label_votes = obj.get("label_votes") or {}
        return [
            str(label)
            for label, _score in sorted(
                label_votes.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
        ]

    def _should_merge_cross_label(self, obj, pos, size):
        obj_center = obj.get("aabb_center", obj.get("coord", [0.0, 0.0, 0.0]))
        obj_size = obj.get("aabb_size", [0.0, 0.0, 0.0])
        overlap = self._aabb_overlap_ratio(
            obj_center,
            obj_size,
            [pos["x"], pos["y"], pos["z"]],
            [size["x"], size["y"], size["z"]],
        )
        return overlap >= 0.6

    def _aabb_overlap_ratio(self, center_a, size_a, center_b, size_b):
        mins_a = [float(center_a[i]) - 0.5 * max(float(size_a[i]), 0.02) for i in range(3)]
        maxs_a = [float(center_a[i]) + 0.5 * max(float(size_a[i]), 0.02) for i in range(3)]
        mins_b = [float(center_b[i]) - 0.5 * max(float(size_b[i]), 0.02) for i in range(3)]
        maxs_b = [float(center_b[i]) + 0.5 * max(float(size_b[i]), 0.02) for i in range(3)]
        inter = 1.0
        vol_a = 1.0
        vol_b = 1.0
        for axis in range(3):
            inter_axis = max(0.0, min(maxs_a[axis], maxs_b[axis]) - max(mins_a[axis], mins_b[axis]))
            inter *= inter_axis
            vol_a *= max(0.02, maxs_a[axis] - mins_a[axis])
            vol_b *= max(0.02, maxs_b[axis] - mins_b[axis])
        union = max(vol_a + vol_b - inter, 1e-6)
        iou = inter / union
        contained = inter / max(min(vol_a, vol_b), 1e-6)
        return max(iou, contained)

    def _has_configured_tracking_memory(self, obj):
        if self.persistent_classes is None:
            return False
        label = normalize_label(
            obj.get("m1_retention_label") or obj.get("m1_canonical_label")
            or obj.get("semantic_name")
        )
        return bool(
            label in self.persistent_classes
            and int(obj.get("observation_count", 0)) >= 2
            and int(obj.get("max_consecutive_observations", 0))
            >= max(2, self._required_confirmations(label))
            and (not self.persistence_requires_m1 or obj.get("m1_confirmed"))
        )

    def _purge_stale(self, now):
        if self.stale_after_sec <= 0.0:
            return
        persistent_labels = {
            "door",
            "portal",
            "gate",
            "fridge",
            "refrigerator",
            "freezer",
            "locker",
            "safe",
            "water_dispenser",
            "dispenser",
        }
        retained = []
        for obj in self.objects:
            age_ok = now - obj.get("last_seen", now) <= self.stale_after_sec
            labels = {
                normalize_label(obj.get("semantic_name")),
                normalize_label(obj.get("m1_canonical_label")),
            }
            if self.persistent_classes is not None:
                persistent = self._has_configured_tracking_memory(obj)
            else:
                persistent = bool(
                    obj.get("is_confirmed")
                    and obj.get("m1_confirmed")
                    and labels.intersection(persistent_labels)
                )
            if age_ok or persistent:
                retained.append(obj)
        changed = len(retained) != len(self.objects)
        self.objects = retained
        if changed:
            self._invalidate_match_indexes()


class SceneGridStore:
    def __init__(self, unknown_id=-1, confidence_step=5):
        self.unknown_id = int(unknown_id)
        self.confidence_step = int(confidence_step)
        self.info = None
        self.scene_data = []
        self.confidence_data = []

    def initialize_from_occupancy_grid(self, occ_grid):
        self.info = occ_grid.info
        size = int(self.info.width * self.info.height)
        if len(self.scene_data) != size:
            self.scene_data = [self.unknown_id] * size
            self.confidence_data = [-1] * size

    def update_cells(self, world_points, scene_id):
        if self.info is None or scene_id is None or scene_id < 0:
            return
        for x, y in world_points:
            coords = world_to_grid(x, y, self.info)
            if coords is None:
                continue
            idx = grid_index(coords[0], coords[1], self.info.width)
            if self.scene_data[idx] == scene_id:
                old = 0 if self.confidence_data[idx] < 0 else self.confidence_data[idx]
                self.confidence_data[idx] = min(100, old + self.confidence_step)
            else:
                self.scene_data[idx] = int(scene_id)
                self.confidence_data[idx] = max(self.confidence_step, 1)
