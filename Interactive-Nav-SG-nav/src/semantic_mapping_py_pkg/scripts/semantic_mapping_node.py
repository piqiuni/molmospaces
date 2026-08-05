#!/usr/bin/env python3
import copy
import json
import math
import os
import threading
import time

import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
import tf2_ros
from geometry_msgs.msg import Point
from geometry_msgs.msg import TransformStamped
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping_py_pkg.geometry_utils import normalize_label, transform_point_best_effort
from semantic_mapping_py_pkg.graph_rules import observation_from_detection
from semantic_mapping_py_pkg.graph_ablation import apply_module1_ablation
from semantic_mapping_py_pkg.interaction_graph_viz import build_graph_marker_array
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.interaction_result_contract import (
    merge_interaction_result_with_command,
    take_pending_interaction_command,
)
from semantic_mapping_py_pkg.messages import dumps_compact, parse_json_list, parse_json_object_or_text
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter, RoomSegmentationState
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore, SceneGridStore
from semantic_mapping_py_pkg.semantic_occ_overlay import OverlayUpdateRegionTracker, SemanticOccupancyOverlay
from semantic_mllm_py_pkg.ablation import AblationConfig


class SemanticMappingNode:
    def __init__(self):
        patch_roslogging_findcaller_for_py311()
        rospy.init_node("semantic_mapping_py")
        topics = get_topics(rospy)
        frames = get_frames(rospy)
        config = get_nested_param(rospy, "semantic_map", {}) or {}
        ablation_config = get_nested_param(rospy, "ablation", {}) or {}
        self.ablation = AblationConfig(
            module1=str(ablation_config.get("module1", "dynamic_rule")),
            module2=str(ablation_config.get("module2", "rule_cost")),
            module3=str(ablation_config.get("module3", "rule_verified")),
        )
        scene_types = get_nested_param(rospy, "scene_types", {}) or {}
        object_room_priors = get_nested_param(rospy, "object_room_priors", {}) or {}
        room_inference_config = get_nested_param(rospy, "room_inference", {}) or {}
        room_mllm_config = get_nested_param(rospy, "room_mllm", {}) or {}

        self.world_frame = frames.get("world_frame", "tf_frame_map")
        self.object_detection_topic = topics.get("object_detections", "/semantic_mapping/object_detections")
        self.scene_attribute_topic = topics.get("scene_attribute", "/semantic_mapping/scene_attribute")
        self.pointcloud_topic = topics.get("pointcloud", "/registered_scan")
        self.occupancy_grid_topic = topics.get("occupancy_grid", "/struct_mapping/occ_map")
        self.room_context_topic = topics.get("room_context", "/semantic_mapping/room_context")
        self.gt_observations_topic = topics.get("gt_observations", "/semantic_mapping/gt_observations")
        self.interaction_command_topic = topics.get(
            "interaction_command", "/semantic_decision/interaction_command"
        )
        self.interaction_result_topic = topics.get("interaction_result", "/semantic_mapping/interaction_result")
        self.attribute_updates_topic = topics.get(
            "attribute_updates", "/semantic_mapping/attribute_updates"
        )
        self.room_attribute_requests_topic = topics.get(
            "room_attribute_requests", "/semantic_mapping/room_attribute_requests"
        )
        self.room_attribute_updates_topic = topics.get(
            "room_attribute_updates", "/semantic_mapping/room_attribute_updates"
        )
        self.planning_occupancy_grid_topic = topics.get(
            "planning_occupancy_grid", "/semantic_mapping/planning_occ_map"
        )
        self.planning_occupancy_grid_updates_topic = topics.get(
            "planning_occupancy_grid_updates", self.planning_occupancy_grid_topic + "_updates"
        )
        self.door_clear_mask_topic = topics.get("door_clear_mask", "/semantic_mapping/door_clear_mask")

        self.object_map_topic = topics.get("object_map", "/semantic_mapping/obj_map")
        self.object_markers_topic = topics.get("object_markers", "/semantic_mapping/object_semantic_map_markers")
        self.scene_id_grid_topic = topics.get("scene_id_grid", "/semantic_mapping/scene_id_grid")
        self.scene_confidence_grid_topic = topics.get("scene_confidence_grid", "/semantic_mapping/scene_confidence_grid")
        self.room_segment_grid_topic = topics.get(
            "room_segment_grid", "/semantic_mapping/room_segment_grid"
        )
        self.unified_graph_topic = topics.get("unified_graph", "/semantic_mapping/unified_graph")
        self.navigation_hints_topic = topics.get("navigation_hints", "/semantic_mapping/navigation_hints")
        self.unified_graph_markers_topic = topics.get(
            "unified_graph_markers", "/semantic_mapping/unified_graph_markers"
        )
        self.unified_graph_markers_lifted_topic = topics.get(
            "unified_graph_markers_lifted", "/semantic_mapping/unified_graph_markers_lifted"
        )

        self.enable_object_mapping = bool(config.get("enable_object_mapping", True))
        self.enable_scene_mapping = bool(config.get("enable_scene_mapping", True))
        self.publish_rate = float(config.get("publish_rate", 2.0))
        self.object_stale_after_sec = float(config.get("object_stale_after_sec", 0.0))
        self.scene_min_range = float(config.get("scene_min_range", 0.1))
        self.scene_max_range = float(config.get("scene_max_range", 3.0))
        self.room_free_threshold = int(config.get("room_free_threshold", 20))
        self.room_unknown_id = int(config.get("room_unknown_id", -1))
        self.room_box_height = float(config.get("room_box_height", 0.2))
        self.room_min_component_cells = int(config.get("room_min_component_cells", 25))
        self.room_boundary_margin_cells = max(0, int(config.get("room_boundary_margin_cells", 1)))
        self.room_core_min_component_cells = max(
            self.room_min_component_cells,
            int(config.get("room_core_min_component_cells", 60)),
        )
        self.room_core_clearance_cells = max(1, int(config.get("room_core_clearance_cells", 7)))
        self.room_small_obstacle_max_cells = max(0, int(config.get("room_small_obstacle_max_cells", 0)))
        self.room_remove_enclosed_occupied = bool(config.get("room_remove_enclosed_occupied", True))
        self.room_enclosed_occupied_max_cells = max(0, int(config.get("room_enclosed_occupied_max_cells", 700)))
        self.room_enclosed_occupied_max_aspect = float(config.get("room_enclosed_occupied_max_aspect", 2.5))
        self.room_enclosed_occupied_known_ring_ratio = float(
            config.get("room_enclosed_occupied_known_ring_ratio", 0.95)
        )
        self.room_enclosed_occupied_free_ring_ratio = float(
            config.get("room_enclosed_occupied_free_ring_ratio", 0.45)
        )
        self.room_portal_cut_enabled = bool(config.get("room_portal_cut_enabled", True))
        self.room_portal_cut_margin_m = float(config.get("room_portal_cut_margin_m", 0.15))
        self.room_portal_cut_thickness_cells = int(config.get("room_portal_cut_thickness_cells", 2))
        self.room_portal_detector_min_confirmations = int(
            config.get("room_portal_detector_min_confirmations", 3)
        )
        self.room_portal_detector_max_center_jump_m = float(
            config.get("room_portal_detector_max_center_jump_m", 0.4)
        )
        self.room_portal_hint_merge_distance_m = float(
            config.get("room_portal_hint_merge_distance_m", 0.6)
        )
        self.room_portal_min_width_m = float(config.get("room_portal_min_width_m", 0.5))
        self.room_portal_max_width_m = float(config.get("room_portal_max_width_m", 2.5))
        self.room_id_overlap_ratio = float(config.get("room_id_overlap_ratio", 0.25))
        self.room_merge_confirmations = int(config.get("room_merge_confirmations", 3))
        self.room_grid_stability_frames = int(
            config.get("room_grid_stability_frames", 3)
        )
        self.room_segment_use_semantic_overlay = bool(
            config.get("room_segment_use_semantic_overlay", True)
        )
        self.room_post_open_force_refresh = bool(
            config.get("room_post_open_force_refresh", True)
        )
        self.room_geometry_stability_frames = int(
            config.get("room_geometry_stability_frames", 5)
        )
        self.room_mllm_enabled = bool(room_mllm_config.get("enabled", True))
        self.room_mllm_min_evidence_objects = max(
            1, int(room_mllm_config.get("min_evidence_objects", 1))
        )
        self.room_mllm_min_confidence = max(
            0.0,
            min(1.0, float(room_mllm_config.get("min_confidence", 0.55))),
        )
        # When enabled, the mapper readiness stream is a causal watermark,
        # rather than a heartbeat: the source OCC frame must have completed
        # room segmentation and the corresponding unified graph publication
        # before ``ready=true`` is emitted.  Keep the switch configurable so
        # lightweight detector-only deployments can retain the old heartbeat
        # contract while the strict smoke test opts in explicitly.
        self.step_ready_require_room_graph = bool(
            config.get("step_ready_require_room_graph", False)
        )
        self.lifted_graph_frame = str(config.get("lifted_graph_frame", "tf_frame_map_graph"))
        self.lifted_graph_z_offset = float(config.get("lifted_graph_z_offset", 10.0))
        self.graph_min_observations = max(1, int(config.get("graph_min_observations", 1)))
        self.graph_save_path = str(config.get("graph_save_path", "") or "").strip()
        self.graph_save_dir = str(config.get("graph_save_dir", "") or "").strip()
        self.graph_save_pretty = bool(config.get("graph_save_pretty", True))
        graph_config = get_nested_param(rospy, "interaction_graph", {}) or {}
        overlay_config = get_nested_param(rospy, "semantic_occ_overlay", {}) or {}
        self.class_to_id = {
            normalize_label(name): int(value)
            for name, value in (scene_types.get("class_to_id", {}) or {}).items()
        }
        self.id_to_class = {int(value): normalize_label(name) for name, value in self.class_to_id.items()}
        self.synonyms = {
            normalize_label(src): normalize_label(dst)
            for src, dst in (scene_types.get("synonyms", {}) or {}).items()
        }

        self.object_store = ObjectMapStore(
            match_distance=config.get("object_match_distance", 0.5),
            stale_after_sec=self.object_stale_after_sec,
            min_confirmations=config.get("object_min_confirmations", 2),
            size_match_ratio=config.get("object_size_match_ratio", 0.7),
            stable_history_size=config.get("object_stable_history_size", 5),
        )
        self.scene_store = SceneGridStore(
            unknown_id=scene_types.get("unknown_id", -1),
            confidence_step=config.get("scene_confidence_step", 5),
        )
        self.graph_store = InteractionGraphStore(
            scene_id=graph_config.get("scene_id", rospy.get_name().strip("/") or "semantic_mapping_scene"),
            match_distance=graph_config.get("match_distance", config.get("object_match_distance", 0.5)),
            room_id_to_name={},
            room_box_height=self.room_box_height,
            portal_room_max_radius_m=graph_config.get(
                "portal_room_max_radius_m", 1.0
            ),
            object_room_search_margin_m=graph_config.get(
                "object_room_search_margin_m", 0.75
            ),
            object_room_priors=object_room_priors,
            room_attribute_min_confidence=room_inference_config.get(
                "min_confidence", 0.2
            ),
            room_mllm_min_confidence=self.room_mllm_min_confidence,
            interaction_geometry_overrides=graph_config.get(
                "geometry_overrides", {}
            ),
        )
        self.semantic_occ_overlay = SemanticOccupancyOverlay(
            enabled=overlay_config.get("enabled", True),
            clear_padding_m=overlay_config.get("clear_padding_m", 0.10),
            open_states=overlay_config.get("open_states", ["open"]),
        )
        # Room segmentation consumes the same graph snapshot as the planning
        # overlay, but it must not share the mutable planning-overlay instance:
        # the room worker runs outside ``self.lock`` while the publishing path
        # continues to update the planning overlay under that lock.
        self._room_segmentation_overlay = SemanticOccupancyOverlay(
            enabled=overlay_config.get("enabled", True),
            clear_padding_m=overlay_config.get("clear_padding_m", 0.10),
            open_states=overlay_config.get("open_states", ["open"]),
        )
        self.semantic_occ_update_tracker = OverlayUpdateRegionTracker()
        # Planning overlay state is shared by interaction callbacks and the
        # timer/graph publishers.  It has its own lock so the expensive map
        # materialization never requires the mapper data lock.
        self._planning_overlay_lock = threading.RLock()
        self._planning_clear_mask_initialized = False
        self._planning_clear_mask_geometry_key = None
        self._planning_overlay_was_active = False

        self.lock = threading.Lock()
        # ``room_segmenter`` has temporal state (stable IDs, portal hints and
        # merge confirmation).  Keep that state serial, but deliberately keep
        # it separate from the mapper's short-lived data/graph lock.
        self._room_lock = threading.RLock()
        self._room_work_condition = threading.Condition()
        self._room_work_pending = None
        self._room_worker_stopping = False
        self._room_worker_thread = None
        self._room_input_revision = 0
        # Raw OCC updates do not invalidate an in-flight room segmentation when
        # the grid geometry is unchanged.  Portal/episode structure changes do.
        self._room_topology_revision = 0
        self._room_epoch = 0
        self._room_last_committed_revision = -1
        # Causal readiness watermark for the latest room/graph commit.  The
        # source identity is copied from the raw OCC message (seq + stamp), so
        # a room result from an older map cannot unlock a newer simulator step.
        self._room_last_commit = None
        self._latest_occupancy_received_mono_s = 0.0
        self._last_causal_ready_source = None
        self._timing_counts = {}
        self._timing_windows = {}
        self._timing_log_every = max(1, int(config.get("timing_log_every", 20)))
        self.latest_cloud = None
        self.latest_scene = None
        self.latest_occupancy_grid = None
        self.latest_room_segment_grid = None
        # Scene grids are large (typically 1984²). Publish them only when the
        # sparse semantic store changes instead of rebuilding two full arrays
        # on every planning-map timer tick.
        self._scene_grid_revision = 0
        self._scene_grid_published_revision = -1
        # A successful portal opening is followed by one immediate planning
        # map publication from the first newer raw OCC.  The normal timer is
        # intentionally not accelerated for every SLAM frame: this narrow
        # hand-off gives the post-open causal gate a source-matched map
        # without turning full-map publishing into the hot path.
        self._post_open_planning_refresh_after_stamp_sec = None
        self._post_open_room_refresh_result = None
        self.pending_interaction_commands = {}
        self.max_pending_interaction_commands = 128
        self.room_segmenter = RoomSegmenter(
            room_free_threshold=self.room_free_threshold,
            room_unknown_id=self.room_unknown_id,
            room_min_component_cells=self.room_min_component_cells,
            room_boundary_margin_cells=self.room_boundary_margin_cells,
            room_core_min_component_cells=self.room_core_min_component_cells,
            room_core_clearance_cells=self.room_core_clearance_cells,
            room_small_obstacle_max_cells=self.room_small_obstacle_max_cells,
            room_remove_enclosed_occupied=self.room_remove_enclosed_occupied,
            room_enclosed_occupied_max_cells=self.room_enclosed_occupied_max_cells,
            room_enclosed_occupied_max_aspect=self.room_enclosed_occupied_max_aspect,
            room_enclosed_occupied_known_ring_ratio=self.room_enclosed_occupied_known_ring_ratio,
            room_enclosed_occupied_free_ring_ratio=self.room_enclosed_occupied_free_ring_ratio,
            room_portal_cut_enabled=self.room_portal_cut_enabled,
            room_portal_cut_margin_m=self.room_portal_cut_margin_m,
            room_portal_cut_thickness_cells=self.room_portal_cut_thickness_cells,
            room_portal_detector_min_confirmations=self.room_portal_detector_min_confirmations,
            room_portal_detector_max_center_jump_m=self.room_portal_detector_max_center_jump_m,
            room_portal_hint_merge_distance_m=self.room_portal_hint_merge_distance_m,
            room_portal_min_width_m=self.room_portal_min_width_m,
            room_portal_max_width_m=self.room_portal_max_width_m,
            room_id_overlap_ratio=self.room_id_overlap_ratio,
            room_merge_confirmations=self.room_merge_confirmations,
            room_grid_stability_frames=self.room_grid_stability_frames,
            state=RoomSegmentationState(),
        )
        self.tf_listener = tf.TransformListener()

        self.object_sub = rospy.Subscriber(self.object_detection_topic, String, self.object_callback, queue_size=10)
        self.scene_sub = rospy.Subscriber(self.scene_attribute_topic, String, self.scene_callback, queue_size=10)
        self.cloud_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.occ_sub = rospy.Subscriber(self.occupancy_grid_topic, OccupancyGrid, self.occupancy_callback, queue_size=1)
        self.room_context_sub = rospy.Subscriber(self.room_context_topic, String, self.room_context_callback, queue_size=1)
        self.gt_observation_sub = rospy.Subscriber(
            self.gt_observations_topic,
            String,
            self.gt_observation_callback,
            queue_size=1,
        )
        self.interaction_command_sub = rospy.Subscriber(
            self.interaction_command_topic, String, self.interaction_command_callback, queue_size=2
        )
        self.interaction_result_sub = rospy.Subscriber(
            self.interaction_result_topic, String, self.interaction_result_callback, queue_size=2
        )
        self.attribute_updates_sub = rospy.Subscriber(
            self.attribute_updates_topic, String, self.attribute_updates_callback, queue_size=2
        )
        self.room_attribute_updates_sub = rospy.Subscriber(
            self.room_attribute_updates_topic,
            String,
            self.room_attribute_updates_callback,
            queue_size=2,
        )

        self.step_ready_pub = rospy.Publisher("/semantic_decision/ready/semantic_mapping", String, queue_size=32)
        self.object_pub = rospy.Publisher(self.object_map_topic, String, queue_size=1)
        self.marker_pub = rospy.Publisher(self.object_markers_topic, MarkerArray, queue_size=1)
        self.scene_id_pub = rospy.Publisher(self.scene_id_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.scene_conf_pub = rospy.Publisher(self.scene_confidence_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.room_segment_pub = rospy.Publisher(
            self.room_segment_grid_topic, OccupancyGrid, queue_size=1, latch=True
        )
        self.unified_graph_pub = rospy.Publisher(self.unified_graph_topic, String, queue_size=1, latch=True)
        self.navigation_hints_pub = rospy.Publisher(self.navigation_hints_topic, String, queue_size=1, latch=True)
        self.room_attribute_requests_pub = rospy.Publisher(
            self.room_attribute_requests_topic, String, queue_size=1
        )
        self.planning_occupancy_grid_pub = rospy.Publisher(
            self.planning_occupancy_grid_topic, OccupancyGrid, queue_size=1, latch=True
        )
        self.planning_occupancy_grid_updates_pub = rospy.Publisher(
            self.planning_occupancy_grid_updates_topic,
            OccupancyGridUpdate,
            queue_size=1,
        )
        self.door_clear_mask_pub = rospy.Publisher(
            self.door_clear_mask_topic, OccupancyGrid, queue_size=1, latch=True
        )
        self.unified_graph_markers_pub = rospy.Publisher(
            self.unified_graph_markers_topic, MarkerArray, queue_size=1, latch=True
        )
        self.unified_graph_markers_lifted_pub = rospy.Publisher(
            self.unified_graph_markers_lifted_topic, MarkerArray, queue_size=1, latch=True
        )
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_callback)
        self.save_graph_service = rospy.Service("/semantic_mapping/save_graph", Trigger, self.save_graph_callback)
        self._publish_lifted_graph_tf()
        self._start_room_worker()

        rospy.loginfo("[semantic_mapping_node.py] object_in=%s scene_in=%s cloud=%s occ=%s",
                      self.object_detection_topic, self.scene_attribute_topic,
                      self.pointcloud_topic, self.occupancy_grid_topic)

    def _start_room_worker(self):
        """Start the coalescing room-segmentation worker.

        Room segmentation is intentionally latest-only: an old grid is not
        useful once a newer SLAM grid is available, and letting those jobs queue
        would make the visual room topology increasingly stale.
        """

        worker = threading.Thread(
            target=self._room_worker_loop,
            name="semantic_room_segmentation",
            daemon=True,
        )
        self._room_worker_thread = worker
        worker.start()
        try:
            rospy.on_shutdown(self._stop_room_worker)
        except AttributeError:
            # Lightweight unit-test rospy stubs do not always expose the
            # lifecycle hook; the worker is daemonised in that case as well.
            pass

    def _stop_room_worker(self):
        condition = getattr(self, "_room_work_condition", None)
        if condition is None:
            return
        with condition:
            self._room_worker_stopping = True
            condition.notify_all()

    def _mark_room_inputs_dirty_locked(self, *, structural=False):
        """Record fresh room input and optionally invalidate topology work.

        A room decomposition is intentionally lower-rate than raw OCC.  New
        occupancy samples are coalesced but may reuse an in-flight same-topology
        segmentation; portal state/hints and episode changes must cancel it.
        """

        self._room_input_revision = int(getattr(self, "_room_input_revision", 0)) + 1
        if structural:
            self._room_topology_revision = int(
                getattr(self, "_room_topology_revision", 0)
            ) + 1
        return self._room_input_revision

    def _reset_room_worker_state(self):
        """Reset worker-owned temporal state outside the mapper data lock."""

        room_lock = getattr(self, "_room_lock", None)
        if room_lock is None:
            self.room_segmenter.state = RoomSegmentationState()
            overlay = getattr(self, "_room_segmentation_overlay", None)
            if overlay is not None:
                overlay.reset()
            return
        with room_lock:
            self.room_segmenter.state = RoomSegmentationState()
            self._room_segmentation_overlay.reset()

    def _update_room_portal_hints(self, observations, *, source_mode, refresh_active=False):
        """Mutate RoomSegmenter state without holding ``self.lock``."""

        room_lock = getattr(self, "_room_lock", None)
        if room_lock is None:
            return self.room_segmenter.update_portal_hints(
                observations,
                source_mode=source_mode,
                refresh_active=refresh_active,
            )
        with room_lock:
            return self.room_segmenter.update_portal_hints(
                observations,
                source_mode=source_mode,
                refresh_active=refresh_active,
            )

    def _run_sync_room_refresh_for_compat(self, *, force_stable=False):
        """Run the legacy synchronous path for minimal test-node doubles."""

        if not force_stable:
            return self._refresh_room_grid_locked()
        try:
            return self._refresh_room_grid_locked(force_stable=True)
        except TypeError as exc:
            # Existing focused tests often replace the legacy method with a
            # no-argument lambda.  Do not make the async migration require
            # every such double to implement the new optional keyword.
            if "force_stable" not in str(exc):
                raise
            return self._refresh_room_grid_locked()

    def _enqueue_room_refresh(
        self,
        *,
        force_stable=False,
        post_open_result=None,
        reason="occupancy",
        source=None,
    ):
        """Coalesce a room refresh request without retaining stale grids.

        The compatibility fallback is useful for the small node doubles used by
        unit tests and for callers that construct a node without ``__init__``.
        Production nodes always use the background worker.
        """

        condition = getattr(self, "_room_work_condition", None)
        worker = getattr(self, "_room_worker_thread", None)
        if condition is None or worker is None:
            if post_open_result is not None:
                return self._refresh_confirmed_open_portal_room_grid_locked(post_open_result)
            self._run_sync_room_refresh_for_compat(force_stable=force_stable)
            return True

        with condition:
            if self._room_worker_stopping:
                return False
            if source is None:
                with self.lock:
                    source = self._occupancy_source_identity(
                        getattr(self, "latest_occupancy_grid", None)
                    )
            current_revision = int(getattr(self, "_room_input_revision", 0))
            current_topology_revision = int(
                getattr(self, "_room_topology_revision", 0)
            )
            current_epoch = int(getattr(self, "_room_epoch", 0))
            pending = self._room_work_pending
            post_open_results = []
            reasons = {str(reason)} if reason else set()
            if pending is not None:
                post_open_results.extend(pending.get("post_open_results") or [])
                reasons.update(pending.get("reasons") or [])
                force_stable = bool(force_stable or pending.get("force_stable"))
            if post_open_result is not None:
                post_open_results.append(dict(post_open_result))
            self._room_work_pending = {
                "input_revision": current_revision,
                "topology_revision": current_topology_revision,
                "epoch": current_epoch,
                "force_stable": bool(force_stable),
                "post_open_results": post_open_results,
                "reasons": reasons,
                # Keep the source requested by the latest enqueue alongside
                # the revision.  The worker may coalesce to an even newer raw
                # grid; it records that transition instead of silently
                # pretending it processed the old source.
                "requested_source": dict(source or {}),
            }
            condition.notify()
        return True

    def _room_worker_loop(self):
        while True:
            with self._room_work_condition:
                while (
                    self._room_work_pending is None
                    and not self._room_worker_stopping
                ):
                    self._room_work_condition.wait()
                if self._room_worker_stopping:
                    return
                request = self._room_work_pending
                self._room_work_pending = None
            try:
                self._process_room_refresh_request(request)
            except Exception as exc:
                if not rospy.is_shutdown():
                    rospy.logerr(
                        "[semantic_mapping_node.py] room worker failed: %s",
                        exc,
                    )

    def _room_request_is_current(
        self,
        *,
        input_revision=None,
        topology_revision=None,
        epoch,
    ):
        """Cancel only topology/episode-invalid room jobs.

        ``input_revision`` remains accepted for old test doubles; production
        requests carry ``topology_revision`` so ordinary same-geometry OCC
        updates do not starve room segmentation.
        """

        with self.lock:
            if int(epoch) != int(self._room_epoch):
                return False
            if topology_revision is not None:
                return int(topology_revision) == int(
                    getattr(self, "_room_topology_revision", 0)
                )
            return int(input_revision) == int(self._room_input_revision)

    def _process_room_refresh_request(self, request):
        """Build a room grid outside ``self.lock`` and commit it by revision."""

        total_t0 = time.perf_counter()
        snapshot_t0 = time.perf_counter()
        requested_source = dict(request.get("requested_source") or {})
        with self.lock:
            if int(request.get("epoch", -1)) != int(self._room_epoch):
                return
            raw = self.latest_occupancy_grid
            if raw is None:
                return
            input_revision = int(self._room_input_revision)
            occupancy_received_mono_s = float(
                getattr(self, "_latest_occupancy_received_mono_s", 0.0)
            )
            topology_revision = int(getattr(self, "_room_topology_revision", 0))
            epoch = int(self._room_epoch)
            graph_payload = apply_module1_ablation(
                self.graph_store.as_graph_dict(), self.ablation.module1
            )
            post_open_references = []
            for result in request.get("post_open_results") or []:
                reference = self._confirmed_open_portal_reference_locked(result)
                if reference is not None:
                    post_open_references.append(reference)
        snapshot_ms = (time.perf_counter() - snapshot_t0) * 1000.0

        # Preserve post-open portal geometry even when the accompanying map
        # job was superseded; the next latest-only job still needs that hint.
        if post_open_references:
            with self._room_lock:
                for reference in post_open_references:
                    self.room_segmenter.update_portal_hints(
                        [reference],
                        source_mode="realtime_gt_observation",
                        refresh_active=True,
                    )
        if not self._room_request_is_current(
            input_revision=input_revision,
            topology_revision=topology_revision,
            epoch=epoch,
        ):
            self._record_component_timing("room_worker_stale_pre_overlay", 0.0)
            return

        overlay_t0 = time.perf_counter()
        with self._room_lock:
            room_occupancy = self._room_segmentation_occupancy_from_snapshot(
                raw,
                graph_payload,
            )
        overlay_ms = (time.perf_counter() - overlay_t0) * 1000.0
        if not self._room_request_is_current(
            input_revision=input_revision,
            topology_revision=topology_revision,
            epoch=epoch,
        ):
            self._record_component_timing("room_worker_stale_pre_segment", 0.0)
            return

        segment_t0 = time.perf_counter()
        with self._room_lock:
            room_ids, room_conf = self._segment_rooms_from_occupancy(
                room_occupancy,
                force_stable=bool(request.get("force_stable")),
            )
            room_merges = self.room_segmenter.consume_confirmed_merges()
        segment_ms = (time.perf_counter() - segment_t0) * 1000.0

        crop_t0 = time.perf_counter()
        room_grid = self._build_cropped_room_segment_grid(room_ids, raw=raw)
        crop_ms = (time.perf_counter() - crop_t0) * 1000.0

        commit_t0 = time.perf_counter()
        commit_lock_t0 = time.perf_counter()
        graph_update_ms = 0.0
        committed = False
        room_commit = None
        with self.lock:
            commit_lock_wait_ms = (time.perf_counter() - commit_lock_t0) * 1000.0
            # New raw OCC alone does not invalidate a same-geometry topology
            # result.  Portal-hint changes and episode resets do, so a room
            # result can make forward progress under a 5 Hz mapping stream.
            if (
                epoch == int(self._room_epoch)
                and topology_revision == int(
                    getattr(self, "_room_topology_revision", 0)
                )
            ):
                self.latest_room_segment_grid = room_grid
                graph_update_t0 = time.perf_counter()
                self.graph_store.update_room_grid(
                    raw.info,
                    room_ids,
                    room_conf,
                    room_merges=room_merges,
                    geometry_stability_frames=self.room_geometry_stability_frames,
                )
                graph_update_ms = (time.perf_counter() - graph_update_t0) * 1000.0
                self._room_last_committed_revision = input_revision
                source = self._occupancy_source_identity(raw)
                try:
                    graph_revision = int(
                        getattr(self.graph_store, "graph_revision", -1)
                    )
                except (TypeError, ValueError):
                    graph_revision = -1
                room_commit = {
                    "source": dict(source or {}),
                    "requested_source": requested_source,
                    "request_source_coalesced": bool(
                        requested_source
                        and not self._same_occupancy_source(
                            requested_source, source
                        )
                    ),
                    "input_revision": input_revision,
                    "topology_revision": topology_revision,
                    "epoch": epoch,
                    "episode_id": str(
                        getattr(self.graph_store, "episode_id", "") or ""
                    ),
                    "graph_revision": graph_revision,
                    "occupancy_received_mono_s": occupancy_received_mono_s,
                    "committed_mono_s": time.monotonic(),
                }
                committed = True
        commit_ms = (time.perf_counter() - commit_t0) * 1000.0
        total_ms = (time.perf_counter() - total_t0) * 1000.0
        self._record_component_timing("room_worker_snapshot", snapshot_ms)
        self._record_component_timing("room_worker_overlay", overlay_ms)
        self._record_component_timing("room_worker_segment", segment_ms)
        self._record_component_timing("room_worker_crop", crop_ms)
        self._record_component_timing("room_worker_commit_lock_wait", commit_lock_wait_ms)
        self._record_component_timing("room_worker_graph_update", graph_update_ms)
        self._record_component_timing("room_worker_commit", commit_ms)
        self._record_component_timing("room_worker_total", total_ms)
        if room_commit is not None:
            room_commit["worker_total_ms"] = total_ms
            room_commit["room_segment_ms"] = segment_ms
            room_commit["graph_update_ms"] = graph_update_ms
            with self.lock:
                # The worker commit itself is the causal hand-off.  Keep a
                # copy under the mapper lock so the next published bundle can
                # prove that its raw source and graph revision are paired.
                if (
                    int(room_commit["epoch"]) == int(self._room_epoch)
                    and int(room_commit["topology_revision"])
                    == int(getattr(self, "_room_topology_revision", 0))
                ):
                    self._room_last_commit = dict(room_commit)
        if not committed:
            self._record_component_timing("room_worker_stale_discard", 0.0)

    @staticmethod
    def _occupancy_source_identity(grid):
        """Return the stable source identity carried by an OCC-like message.

        GMapping and the semantic products preserve the input header sequence
        and stamp.  Sequence alone is not sufficient for a few ROS publishers
        that leave it at zero, so the readiness contract prefers a non-zero
        timestamp and falls back to the sequence when a timestamp is absent.
        """

        if grid is None:
            return None
        header = getattr(grid, "header", None)
        if header is None:
            return None
        try:
            step_index = int(getattr(header, "seq", -1))
        except (TypeError, ValueError):
            step_index = -1
        stamp = getattr(header, "stamp", None)
        try:
            stamp_sec = float(stamp.to_sec()) if stamp is not None else 0.0
        except (AttributeError, TypeError, ValueError):
            stamp_sec = 0.0
        return {"step_index": step_index, "stamp_sec": stamp_sec}

    @staticmethod
    def _same_occupancy_source(left, right):
        """Whether two source identities denote the same raw OCC frame."""

        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        try:
            left_stamp = float(left.get("stamp_sec", 0.0) or 0.0)
            right_stamp = float(right.get("stamp_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            left_stamp = right_stamp = 0.0
        if left_stamp > 0.0 and right_stamp > 0.0:
            return abs(left_stamp - right_stamp) <= 1e-6
        try:
            left_step = int(left.get("step_index", -1))
            right_step = int(right.get("step_index", -1))
        except (TypeError, ValueError):
            return False
        return left_step >= 0 and right_step >= 0 and left_step == right_step

    def _semantic_mapping_ready_payload_for_bundle(self, bundle):
        """Build the mapper readiness watermark for one published bundle.

        The method is deliberately a small interface around the causal
        bookkeeping.  ``publish_callback`` invokes it *after* the bundle has
        been sent, so ``unified_graph_ready`` means that the graph associated
        with the room commit was actually emitted, not merely changed in an
        in-memory store.
        """

        bundle = bundle or {}
        raw = bundle.get("raw_occupancy_grid")
        raw_identity = self._occupancy_source_identity(raw)
        room_grid = bundle.get("room_segment_grid")
        room_commit = bundle.get("room_commit") or {}
        room_identity = room_commit.get("source")
        room_grid_identity = self._occupancy_source_identity(room_grid)
        try:
            room_commit_epoch = int(room_commit.get("epoch", -1))
        except (TypeError, ValueError):
            room_commit_epoch = -1
        raw_occ_ready = raw_identity is not None
        room_seg_ready = bool(
            raw_occ_ready
            and room_grid is not None
            and self._same_occupancy_source(raw_identity, room_identity)
            and self._same_occupancy_source(raw_identity, room_grid_identity)
            and room_commit_epoch == int(getattr(self, "_room_epoch", 0))
        )
        graph_payload = bundle.get("graph_payload") or {}
        try:
            graph_revision = int(graph_payload.get("graph_revision", -1))
        except (TypeError, ValueError):
            graph_revision = -1
        try:
            commit_graph_revision = int(room_commit.get("graph_revision", -1))
        except (TypeError, ValueError):
            commit_graph_revision = -1
        graph_episode = str(graph_payload.get("episode_id") or "")
        commit_episode = str(room_commit.get("episode_id") or "")
        unified_graph_ready = bool(
            room_seg_ready
            and graph_revision >= commit_graph_revision >= 0
            and (not commit_episode or graph_episode == commit_episode)
        )
        strict_ready = bool(
            raw_occ_ready and room_seg_ready and unified_graph_ready
            if getattr(self, "step_ready_require_room_graph", False)
            else raw_occ_ready
        )
        missing = []
        if not raw_occ_ready:
            missing.append("occupancy")
        if getattr(self, "step_ready_require_room_graph", False):
            if not room_seg_ready:
                missing.append("room_segmentation")
            if not unified_graph_ready:
                missing.append("unified_graph")
        received_mono = room_commit.get("occupancy_received_mono_s")
        try:
            received_mono = float(received_mono)
            room_latency_ms = (
                max(0.0, (time.monotonic() - received_mono) * 1000.0)
                if received_mono > 0.0
                else None
            )
        except (TypeError, ValueError):
            room_latency_ms = None
        payload = {
            "module": "semantic_mapping",
            "ready": bool(strict_ready),
            "step_index": int(raw_identity.get("step_index", -1)) if raw_identity else -1,
            "stamp_sec": float(raw_identity.get("stamp_sec", 0.0)) if raw_identity else 0.0,
            "raw_occ_ready": bool(raw_occ_ready),
            "room_segmentation_ready": bool(room_seg_ready),
            "unified_graph_ready": bool(unified_graph_ready),
            "causal_contract": (
                "occ_room_graph_same_source"
                if getattr(self, "step_ready_require_room_graph", False)
                else "occ_only"
            ),
            "required_after_observation": [
                "occupancy",
                "room_segmentation",
                "unified_graph",
            ]
            if getattr(self, "step_ready_require_room_graph", False)
            else ["occupancy"],
            "missing_stages": missing,
            "room_commit_source": dict(room_identity or {}),
            "unified_graph_room_source": dict(room_identity or {}),
            "room_requested_source": dict(room_commit.get("requested_source") or {}),
            "room_request_source_coalesced": bool(
                room_commit.get("request_source_coalesced", False)
            ),
            "occupancy_source": dict(raw_identity or {}),
            "room_segmentation_source": dict(room_grid_identity or {}),
            "room_commit_input_revision": room_commit.get("input_revision", -1),
            "room_commit_graph_revision": commit_graph_revision,
            "published_graph_revision": graph_revision,
            "published_graph_capture_step": graph_payload.get("capture_step"),
            "published_graph_timestamp": graph_payload.get("timestamp"),
            "room_commit_latency_ms": room_latency_ms,
            "room_worker_total_ms": room_commit.get("worker_total_ms"),
            "timestamp": time.time(),
        }
        return payload

    def _record_causal_ready_once(self, payload):
        """Emit one timing/log sample for each strict source watermark."""

        if not (
            getattr(self, "step_ready_require_room_graph", False)
            and bool(payload.get("ready"))
        ):
            return
        source = payload.get("occupancy_source") or {}
        try:
            source_seq = int(source.get("step_index", -1))
        except (TypeError, ValueError):
            source_seq = -1
        try:
            source_stamp = float(source.get("stamp_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            source_stamp = 0.0
        # Header.seq can be regenerated by room/map adapters.  Match the
        # strict aggregate contract: the nonzero shared stamp is canonical;
        # sequence is only a fallback and remains telemetry in the log.
        key = (
            ("stamp", round(source_stamp, 6))
            if source_stamp > 0.0
            else ("seq", source_seq)
        )
        lock = getattr(self, "lock", None)
        if lock is None:
            return
        with lock:
            if key == getattr(self, "_last_causal_ready_source", None):
                return
            self._last_causal_ready_source = key
        latency_ms = payload.get("room_commit_latency_ms")
        if latency_ms is not None:
            self._record_component_timing("step_ready_causal_latency", latency_ms)
        rospy.loginfo(
            "SemanticMappingCausalReady source_seq=%d source_stamp=%.6f "
            "room_seq=%s room_stamp=%s graph_revision=%s capture_step=%s "
            "room_latency_ms=%s worker_total_ms=%s",
            source_seq,
            source_stamp,
            (payload.get("room_segmentation_source") or {}).get("step_index"),
            (payload.get("room_segmentation_source") or {}).get("stamp_sec"),
            payload.get("published_graph_revision"),
            payload.get("published_graph_capture_step"),
            "%.3f" % float(latency_ms) if latency_ms is not None else "missing",
            "%.3f" % float(payload["room_worker_total_ms"])
            if payload.get("room_worker_total_ms") is not None
            else "missing",
        )

    def _room_segmentation_occupancy_from_snapshot(self, raw, graph_payload):
        if raw is None or not self.room_segment_use_semantic_overlay:
            return raw
        self._room_segmentation_overlay.update_graph(graph_payload)
        effective_data, _mask_data, _stats = self._room_segmentation_overlay.apply(
            raw.info,
            raw.data,
            include_pending=False,
        )
        return self._build_occupancy_copy(effective_data, raw=raw)

    def object_callback(self, msg):
        if not self.enable_object_mapping:
            return
        parsed = parse_json_object_or_text(msg.data)
        detections = parsed.get("detections")
        if detections is None:
            detections = parse_json_list(msg.data)
        if not isinstance(detections, list):
            return
        stamp = self._stamp_from_detection_payload(parsed)
        with self.lock:
            self.object_store.update(detections, stamp)
            tracked_detections = self.object_store.as_tracked_detections(
                min_observations=self.graph_min_observations,
                confirmed_only=False,
            )
            observations = [
                observation_from_detection(det, observation_id=f"det_{index:04d}")
                for index, det in enumerate(tracked_detections, start=1)
            ]
            self.graph_store.update_observations(
                observations,
                stamp=stamp,
                source_mode="detector_online",
            )
            self.graph_store.prune_stale_nodes(self.object_stale_after_sec, now=stamp)
        # Portal-hint state is worker-owned.  Updating it can wait behind an
        # in-flight segmentation, but it must never make every mapper callback
        # wait on that segmentation while holding ``self.lock``.
        portal_structure_changed = self._update_room_portal_hints(
            observations,
            source_mode="detector_online",
        )
        if portal_structure_changed:
            with self.lock:
                self._mark_room_inputs_dirty_locked(structural=True)
            self._enqueue_room_refresh(reason="object_portal_hint")
        self._safe_publish_bundle(self._collect_publish_bundle())

    def scene_callback(self, msg):
        if not self.enable_scene_mapping:
            return
        parsed = parse_json_object_or_text(msg.data)
        scene_name = normalize_label(parsed.get("scene_attribute", "unknown"))
        scene_name = self.synonyms.get(scene_name, scene_name)
        scene_id = self.class_to_id.get(scene_name, -1)
        if scene_id < 0:
            return
        with self.lock:
            self.latest_scene = {"name": scene_name, "id": scene_id}
            cloud = self.latest_cloud
        if cloud is not None:
            self._update_scene_from_cloud(cloud, scene_id)

    def gt_observation_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        observations = parsed.get("observations")
        if not isinstance(observations, list):
            return
        capture_step = self._capture_step_from_payload(parsed)
        # ``capture_step`` is envelope metadata in the restricted-GT contract.
        # Copy it into the mapper's private observation records so delayed
        # Module-1 patches can be rejected against the correct version rather
        # than every minimal-GT observation silently looking like frame zero.
        observations = [
            {
                **observation,
                **({"_capture_step": capture_step} if capture_step is not None else {}),
            }
            for observation in observations
            if isinstance(observation, dict)
        ]
        episode_id = str(parsed.get("episode_id") or "")
        stamp_value = parsed.get("stamp_sec")
        stamp = (
            float(stamp_value)
            if stamp_value is not None
            else rospy.Time.now().to_sec()
        )
        episode_reset_requested = False
        portal_structure_changed = False
        with self.lock:
            episode_changed = episode_id and episode_id != self.graph_store.episode_id
            episode_reset_requested = bool(parsed.get("episode_reset")) or bool(episode_changed)
            if episode_reset_requested:
                self._save_episode_graph_locked(final=True)
                self.graph_store.reset(episode_id=episode_id, source_mode="realtime_gt_observation")
                self._post_open_planning_refresh_after_stamp_sec = None
                self._post_open_room_refresh_result = None
                self.pending_interaction_commands.clear()
                self.object_store.objects = []
                self.object_store.next_id = 1
                self._room_epoch = int(getattr(self, "_room_epoch", 0)) + 1
                self._room_last_commit = None
                self._last_causal_ready_source = None
                self._mark_room_inputs_dirty_locked(structural=True)
            self.graph_store.update_observations(
                observations,
                stamp=stamp,
                source_mode="realtime_gt_observation",
                capture_step=capture_step,
            )
        # Reset and portal hints are worker-owned state.  Both operations are
        # deliberately outside the mapper lock so an in-flight segmentation
        # cannot stall incoming OCC or graph callbacks.
        if episode_reset_requested:
            self._reset_room_worker_state()
            self._reset_planning_overlay_state()
        portal_structure_changed = self._update_room_portal_hints(
            observations,
            source_mode="realtime_gt_observation",
        )
        if portal_structure_changed:
            with self.lock:
                self._mark_room_inputs_dirty_locked(structural=True)
        if episode_reset_requested or portal_structure_changed:
            self._enqueue_room_refresh(reason="gt_observation")
        self._safe_publish_bundle(self._collect_publish_bundle())

    def interaction_command_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        opening = str(parsed.get("action") or "").casefold() == "open"
        with self.lock:
            self._remember_interaction_command_locked(parsed)
        changed = self._set_planning_interaction_pending(parsed.get("node_id"), True) if opening else False
        if changed:
            self._safe_publish_bundle(self._collect_publish_bundle())

    def interaction_result_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        deferred_room_refresh_result = None
        pending_node_id = ""
        with self.lock:
            command = self._take_interaction_command_locked(parsed)
            parsed = merge_interaction_result_with_command(parsed, command)
            stamp_value = parsed.get("stamp_sec")
            stamp = (
                float(stamp_value)
                if stamp_value is not None
                else rospy.Time.now().to_sec()
            )
            pending_node_id = str(parsed.get("node_id") or "")
            changed = self.graph_store.update_interaction_result(parsed, stamp=stamp)
            successful_open = self._is_successful_open_result(parsed)
            node_type = str(parsed.get("node_type") or "").casefold()
            portal_reference = (
                self._confirmed_open_portal_reference_locked(parsed)
                if successful_open and node_type != "portal"
                else None
            )
            is_portal_open = successful_open and (
                node_type == "portal"
                or portal_reference is not None
            )
            if is_portal_open:
                # This must not depend on ``changed``.  A result can arrive
                # after the graph has already been updated, while the next
                # raw OCC is still the first map that can prove post-open
                # traversability to the planner.
                self._post_open_planning_refresh_after_stamp_sec = stamp
                self._post_open_room_refresh_result = dict(parsed)
                rospy.loginfo(
                    "[semantic_mapping_node.py] armed post-open raw OCC bridge: "
                    "result_stamp=%.6f node_id=%s node_type=%s "
                    "graph_changed=%s portal_reference=%s",
                    stamp,
                    str(parsed.get("node_id") or ""),
                    str(parsed.get("node_type") or ""),
                    changed,
                    portal_reference is not None,
                )
            elif changed and self.room_post_open_force_refresh:
                # Preserve the old non-portal recovery behavior, but queue the
                # expensive topology rebuild instead of running it in this ROS
                # callback while the mapper lock is held.
                deferred_room_refresh_result = dict(parsed)
                self._mark_room_inputs_dirty_locked(structural=True)
        pending_changed = self._set_planning_interaction_pending(
            pending_node_id, False
        )
        if deferred_room_refresh_result is not None:
            self._enqueue_room_refresh(
                force_stable=True,
                post_open_result=deferred_room_refresh_result,
                reason="interaction_result",
            )
        if (changed or pending_changed) and not is_portal_open:
            self._safe_publish_bundle(self._collect_publish_bundle())

    def _remember_interaction_command_locked(self, command):
        key = str(command.get("command_id") or command.get("event_id") or "")
        if not key:
            return
        self.pending_interaction_commands.pop(key, None)
        self.pending_interaction_commands[key] = dict(command)
        while len(self.pending_interaction_commands) > self.max_pending_interaction_commands:
            oldest_key = next(iter(self.pending_interaction_commands))
            self.pending_interaction_commands.pop(oldest_key, None)

    def _take_interaction_command_locked(self, result):
        return take_pending_interaction_command(
            self.pending_interaction_commands, result
        )

    def attribute_updates_callback(self, msg):
        if self.ablation.module1 != "dynamic_mllm":
            return
        parsed = parse_json_object_or_text(msg.data)
        episode_id = str(parsed.get("episode_id") or "")
        if episode_id and self.graph_store.episode_id and episode_id != self.graph_store.episode_id:
            return
        stamp_value = parsed.get("stamp_sec")
        stamp = (
            float(stamp_value)
            if stamp_value is not None
            else rospy.Time.now().to_sec()
        )
        changed = False
        with self.lock:
            for patch in parsed.get("updates") or []:
                if isinstance(patch, dict):
                    changed = self.graph_store.apply_attribute_patch(patch, stamp=stamp) or changed
        if changed:
            self._safe_publish_bundle(self._collect_publish_bundle())

    def room_attribute_updates_callback(self, msg):
        """Apply room-only Module-1 updates on their dedicated topic."""

        if self.ablation.module1 != "dynamic_mllm":
            return
        parsed = parse_json_object_or_text(msg.data)
        episode_id = str(parsed.get("episode_id") or "")
        if episode_id and self.graph_store.episode_id and episode_id != self.graph_store.episode_id:
            return
        stamp_value = parsed.get("stamp_sec")
        stamp = (
            float(stamp_value)
            if stamp_value is not None
            else rospy.Time.now().to_sec()
        )
        changed = False
        with self.lock:
            for patch in parsed.get("updates") or []:
                if isinstance(patch, dict):
                    changed = (
                        self.graph_store.apply_room_attribute_patch(patch, stamp=stamp)
                        or changed
                    )
        if changed:
            self._safe_publish_bundle(self._collect_publish_bundle())

    def pointcloud_callback(self, msg):
        with self.lock:
            self.latest_cloud = msg

    def room_context_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        rooms = parsed.get("rooms")
        room_id_to_name = parsed.get("room_id_to_name") or {}
        if not isinstance(rooms, list):
            return
        with self.lock:
            if isinstance(room_id_to_name, dict):
                self.graph_store.room_id_to_name.update(
                    {
                        int(room_id): normalize_label(room_name)
                        for room_id, room_name in room_id_to_name.items()
                    }
                )
            self.graph_store.set_room_geometries(rooms)

    def _record_component_timing(self, kind, elapsed_ms):
        # Some focused unit tests construct a minimal node via ``__new__``;
        # timing must remain observational and never become a test/runtime
        # dependency.
        if not hasattr(self, "_timing_counts"):
            self._timing_counts = {}
        if not hasattr(self, "_timing_windows"):
            self._timing_windows = {}
        if not hasattr(self, "_timing_log_every"):
            self._timing_log_every = 20
        count = int(self._timing_counts.get(kind, 0)) + 1
        self._timing_counts[kind] = count
        window = self._timing_windows.setdefault(kind, [])
        window.append(float(elapsed_ms))
        if len(window) > 200:
            del window[:-200]
        if count % self._timing_log_every:
            return
        ordered = sorted(window)
        percentile = lambda q: ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))]
        rospy.loginfo(
            "SemanticMappingTiming kind=%s n=%d window=%d avg/p50/p95/max=%.3f/%.3f/%.3f/%.3fms",
            kind,
            count,
            len(window),
            sum(window) / max(1, len(window)),
            percentile(0.50),
            percentile(0.95),
            max(window),
        )

    def occupancy_callback(self, msg):
        callback_t0 = time.perf_counter()
        lock_t0 = time.perf_counter()
        deferred_room_refresh_result = None
        force_stable = False
        source_identity = self._occupancy_source_identity(msg)
        with self.lock:
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
            self.latest_occupancy_grid = msg
            # Capture receipt time next to the raw source pointer.  The room
            # worker may process a coalesced request later, so a global timing
            # counter would otherwise report the newer frame's latency.
            self._latest_occupancy_received_mono_s = time.monotonic()
            self._mark_room_inputs_dirty_locked()
            post_open_t0 = time.perf_counter()
            if self._raw_occupancy_is_after_post_open_refresh_locked(msg):
                result_stamp = self._post_open_planning_refresh_after_stamp_sec
                # Publish the first source-new raw map before room
                # segmentation.  The room refresh itself is now queued after
                # this short critical section, so incoming raw OCC never waits
                # for the full room pipeline.
                self._publish_post_open_raw_planning_grid_locked(
                    msg,
                    result_stamp=result_stamp,
                )
                self._post_open_planning_refresh_after_stamp_sec = None
                deferred_room_refresh_result = getattr(
                    self, "_post_open_room_refresh_result", None
                )
                self._post_open_room_refresh_result = None
                force_stable = bool(
                    getattr(self, "room_post_open_force_refresh", True)
                )
            post_open_ms = (time.perf_counter() - post_open_t0) * 1000.0
            scene_init_t0 = time.perf_counter()
            scene_data_before = getattr(self.scene_store, "scene_data", None)
            scene_size_before = (
                len(scene_data_before) if scene_data_before is not None else None
            )
            self.scene_store.initialize_from_occupancy_grid(msg)
            scene_data_after = getattr(self.scene_store, "scene_data", None)
            if (
                scene_size_before is not None
                and scene_data_after is not None
                and len(scene_data_after) != scene_size_before
            ):
                self._scene_grid_revision = int(
                    getattr(self, "_scene_grid_revision", 0)
                ) + 1
            scene_init_ms = (time.perf_counter() - scene_init_t0) * 1000.0
        enqueue_t0 = time.perf_counter()
        self._enqueue_room_refresh(
            force_stable=force_stable,
            post_open_result=(
                deferred_room_refresh_result if force_stable else None
            ),
            reason="post_open" if deferred_room_refresh_result is not None else "occupancy",
            source=source_identity,
        )
        room_enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        total_ms = (time.perf_counter() - callback_t0) * 1000.0
        self._record_component_timing("occupancy_lock_wait", lock_wait_ms)
        self._record_component_timing("occupancy_post_open_bridge", post_open_ms)
        self._record_component_timing("occupancy_scene_init", scene_init_ms)
        self._record_component_timing("occupancy_room_enqueue", room_enqueue_ms)
        self._record_component_timing("occupancy_callback_total", total_ms)

    @staticmethod
    def _is_successful_open_result(result):
        return (
            result.get("success") is True
            and str(result.get("action") or "").casefold() == "open"
        )

    def _publish_post_open_raw_planning_grid_locked(self, planning_grid, *, result_stamp):
        """Directly hand the first post-open raw OCC to the planning map."""

        self.planning_occupancy_grid_pub.publish(planning_grid)
        raw_stamp = self._occupancy_header_stamp_sec(planning_grid)
        rospy.loginfo(
            "[semantic_mapping_node.py] published post-open raw planning OCC: "
            "raw_stamp=%s result_stamp=%s raw_seq=%s",
            "%.6f" % raw_stamp if raw_stamp is not None else "missing",
            "%.6f" % result_stamp if result_stamp is not None else "missing",
            int(getattr(planning_grid.header, "seq", 0) or 0),
        )

    def _raw_occupancy_is_after_post_open_refresh_locked(self, msg):
        """Return whether ``msg`` consumes the one-shot portal-open refresh.

        Prefer the source header relationship whenever both publishers provide
        stamps.  If a raw map has no usable stamp, arrival after the mapper's
        interaction-result callback is the only available ordering signal;
        the executor records that fallback explicitly in its causal trace.
        """

        result_stamp = self._post_open_planning_refresh_after_stamp_sec
        if result_stamp is None:
            return False
        raw_stamp = self._occupancy_header_stamp_sec(msg)
        if raw_stamp is None:
            return True
        return raw_stamp > result_stamp

    @staticmethod
    def _occupancy_header_stamp_sec(msg):
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        try:
            value = float(stamp.to_sec())
        except (AttributeError, TypeError, ValueError):
            try:
                value = float(stamp.secs) + float(stamp.nsecs) * 1e-9
            except (AttributeError, TypeError, ValueError):
                return None
        return value if math.isfinite(value) and value > 0.0 else None

    def _refresh_confirmed_open_portal_room_grid_locked(self, result):
        """Commit a real room split immediately after a successful portal open.

        The planner is allowed to use a pending door clear region, while room
        topology is not.  At this point the interaction graph has already
        confirmed the postcondition, so use the immutable pre-open doorway
        reference to refresh its virtual cut and force one stable segmentation
        update.  The graph store will then replace its synthetic portal child
        only when observed free space actually yields a second room ID.
        """

        if result.get("success") is False:
            return False
        action = str(result.get("action") or "").casefold()
        if action and action != "open":
            return False
        reference = self._confirmed_open_portal_reference_locked(result)
        if reference is None:
            return False
        self.room_segmenter.update_portal_hints(
            [reference],
            source_mode="realtime_gt_observation",
            refresh_active=True,
        )
        self._run_sync_room_refresh_for_compat(force_stable=True)
        return True

    def _confirmed_open_portal_reference_locked(self, result):
        """Build a stable, closed-door geometry observation for a result."""

        requested_node_id = str(result.get("node_id") or "")
        requested_object_id = str(
            result.get("object_id") or result.get("instance_id") or ""
        )
        for node in self.graph_store.as_graph_dict().get("nodes") or []:
            if node.get("type") != "portal":
                continue
            attributes = node.get("attributes") or {}
            instance_id = str(attributes.get("instance_id") or "")
            source_object_name = str(attributes.get("source_object_name") or "")
            if (
                requested_node_id
                and str(node.get("id") or "") != requested_node_id
                and (
                    not requested_object_id
                    or (
                        instance_id != requested_object_id
                        and source_object_name != requested_object_id
                    )
                )
            ):
                continue
            if (
                requested_object_id
                and not requested_node_id
                and instance_id != requested_object_id
                and source_object_name != requested_object_id
            ):
                continue
            state = str((node.get("interaction") or {}).get("state") or "").casefold()
            if state not in {"open", "ajar", "static_open"}:
                continue
            center = list(
                attributes.get("interaction_reference_aabb_center")
                or node.get("aabb_center")
                or []
            )
            size = list(
                attributes.get("interaction_reference_aabb_size")
                or node.get("aabb_size")
                or []
            )
            if len(center) < 2 or len(size) < 2:
                continue
            reference_id = str(
                attributes.get("source_object_name")
                or instance_id
                or node.get("id")
            )
            return {
                "id": reference_id,
                "name": "door",
                "is_door": True,
                "box_3d": {"center": center, "size": size},
            }
        return None

    def _room_segmentation_occupancy_locked(self):
        """Return raw occupancy plus confirmed portal clears for room labels."""

        raw = self.latest_occupancy_grid
        if raw is None or not self.room_segment_use_semantic_overlay:
            return raw
        graph_payload = apply_module1_ablation(
            self.graph_store.as_graph_dict(), self.ablation.module1
        )
        self.semantic_occ_overlay.update_graph(graph_payload)
        effective_data, _mask_data, _stats = self.semantic_occ_overlay.apply(
            raw.info,
            raw.data,
            include_pending=False,
        )
        return self._build_occupancy_copy(effective_data)

    def _refresh_room_grid_locked(self, *, force_stable=False):
        if self.latest_occupancy_grid is None:
            return
        room_occupancy = self._room_segmentation_occupancy_locked()
        room_ids, room_conf = self._segment_rooms_from_occupancy(
            room_occupancy,
            force_stable=force_stable,
        )
        room_merges = self.room_segmenter.consume_confirmed_merges()
        self.latest_room_segment_grid = self._build_cropped_room_segment_grid(room_ids)
        self.graph_store.update_room_grid(
            self.latest_occupancy_grid.info,
            room_ids,
            room_conf,
            room_merges=room_merges,
            geometry_stability_frames=self.room_geometry_stability_frames,
        )

    def publish_callback(self, _event):
        if rospy.is_shutdown():
            return
        callback_t0 = time.perf_counter()
        lock_t0 = time.perf_counter()
        with self.lock:
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
        collect_t0 = time.perf_counter()
        publish_bundle = self._collect_publish_bundle()
        collect_ms = (time.perf_counter() - collect_t0) * 1000.0
        publish_t0 = time.perf_counter()
        self._safe_publish_bundle(publish_bundle)
        # Publish readiness only after this exact bundle (including room grid
        # and unified graph) has been emitted.  In strict mode the payload is a
        # causal watermark for one raw OCC source, not a periodic heartbeat.
        ready_payload = self._semantic_mapping_ready_payload_for_bundle(publish_bundle)
        self.step_ready_pub.publish(String(data=json.dumps(ready_payload, separators=(",", ":"))))
        self._record_causal_ready_once(ready_payload)
        publish_ms = (time.perf_counter() - publish_t0) * 1000.0
        total_ms = (time.perf_counter() - callback_t0) * 1000.0
        self._record_component_timing("publish_lock_wait", lock_wait_ms)
        self._record_component_timing("publish_collect_bundle", collect_ms)
        self._record_component_timing("publish_ros_messages", publish_ms)
        self._record_component_timing("publish_callback_total", total_ms)

    def _safe_publish_bundle(self, bundle):
        try:
            self._publish_bundle(bundle)
        except rospy.ROSException as exc:
            if "closed topic" not in str(exc).lower() and not rospy.is_shutdown():
                raise

    def _update_scene_from_cloud(self, cloud, scene_id):
        points = []

        for index, point in enumerate(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if index % 20 != 0:
                continue
            x, y, z = point
            dist = math.sqrt(x * x + y * y + z * z)
            if dist < self.scene_min_range or dist > self.scene_max_range:
                continue
            try:
                (wx, wy, _wz), used_stamp = transform_point_best_effort(
                    self.tf_listener, self.world_frame, cloud.header.frame_id, cloud.header.stamp, (x, y, z)
                )
                if used_stamp == rospy.Time(0):
                    rospy.logwarn_throttle(
                        2.0,
                        "[semantic_mapping_node.py] TF scene cloud fallback to latest transform for %s <- %s",
                        self.world_frame,
                        cloud.header.frame_id,
                    )
            except Exception:
                continue
            points.append((wx, wy))
        if not points:
            return
        with self.lock:
            self.scene_store.update_cells(points, scene_id)
            self._scene_grid_revision = int(
                getattr(self, "_scene_grid_revision", 0)
            ) + 1

    def _build_grid(self, data):
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.world_frame
        grid.info = self.scene_store.info
        grid.data = [int(v) for v in data]
        return grid

    def _build_room_attribute_request_locked(self, graph_payload):
        """Build no-image room evidence for the independent Module-1 lane."""

        if not (
            self.room_mllm_enabled and self.ablation.module1 == "dynamic_mllm"
        ):
            return None
        nodes = list(graph_payload.get("nodes") or [])
        evidence_by_room = {}
        for node in nodes:
            if str(node.get("type") or "") in {"scene", "room", "portal"}:
                continue
            room_id = node.get("room_id")
            if room_id is None:
                continue
            try:
                room_id = int(room_id)
            except (TypeError, ValueError):
                continue
            attributes = node.get("attributes") or {}
            evidence_by_room.setdefault(room_id, []).append(
                {
                    "object_id": str(attributes.get("instance_id") or node.get("id") or ""),
                    "node_id": str(node.get("id") or ""),
                    "name": str(node.get("name") or node.get("label") or "object"),
                    "category": str(attributes.get("category") or ""),
                    "type": str(node.get("type") or "object"),
                    "confidence": float(node.get("confidence", 0.0) or 0.0),
                    "currently_visible": bool(node.get("is_currently_visible", False)),
                }
            )

        rooms = []
        for node in nodes:
            if str(node.get("type") or "") != "room":
                continue
            room_id = node.get("room_id")
            attributes = node.get("attributes") or {}
            try:
                room_id = int(room_id)
            except (TypeError, ValueError):
                continue
            if not attributes.get("active", True) or attributes.get("is_potential_room", False):
                continue
            evidence = sorted(
                evidence_by_room.get(room_id, []),
                key=lambda item: (item["object_id"], item["node_id"]),
            )
            if len(evidence) < self.room_mllm_min_evidence_objects:
                continue
            rooms.append(
                {
                    "room_id": room_id,
                    "room_node_id": str(node.get("id") or f"room_{room_id}"),
                    # This lane intentionally receives no RGB, crop, pose, or
                    # geometric evidence: room ID plus room-member objects is
                    # the complete model context.
                    "objects": evidence,
                }
            )
        if not rooms:
            return None
        capture_step = graph_payload.get("capture_step")
        try:
            capture_step = int(capture_step)
        except (TypeError, ValueError):
            capture_step = None
        return {
            "episode_id": str(graph_payload.get("episode_id") or ""),
            "stamp_sec": float(graph_payload.get("timestamp") or rospy.Time.now().to_sec()),
            "graph_revision": int(graph_payload.get("graph_revision", 0) or 0),
            "capture_step": capture_step,
            "rooms": sorted(rooms, key=lambda item: (item["room_id"], item["room_node_id"])),
        }

    def _snapshot_publish_inputs_locked(self):
        """Capture cheap immutable-or-latest references for a publish build.

        This intentionally does not materialize any full OccupancyGrid data.
        Scene lists and the raw ROS message are stable enough for a visual
        snapshot: occupancy geometry replacement swaps the list object, and
        same-geometry scene updates only touch sparse cells.
        """

        scene_store = getattr(self, "scene_store", None)
        object_store = getattr(self, "object_store", None)
        enable_scene_mapping = bool(getattr(self, "enable_scene_mapping", False))
        enable_object_mapping = bool(getattr(self, "enable_object_mapping", False))
        scene_info_ready = bool(
            enable_scene_mapping
            and scene_store is not None
            and getattr(scene_store, "info", None) is not None
        )
        scene_revision = int(getattr(self, "_scene_grid_revision", 0))
        publish_scene_grids = bool(
            scene_info_ready
            and scene_revision
            != int(getattr(self, "_scene_grid_published_revision", -1))
        )
        graph_payload = apply_module1_ablation(
            self.graph_store.as_graph_dict(), self.ablation.module1
        )
        return {
            "obj_map": (
                object_store.as_obj_map()
                if enable_object_mapping and object_store is not None
                else None
            ),
            "scene_info": getattr(scene_store, "info", None) if scene_info_ready else None,
            "scene_data": getattr(scene_store, "scene_data", None) if scene_info_ready else None,
            "scene_confidence_data": (
                getattr(scene_store, "confidence_data", None)
                if scene_info_ready
                else None
            ),
            "scene_revision": scene_revision,
            "publish_scene_grids": publish_scene_grids,
            "room_segment_grid": self.latest_room_segment_grid,
            "room_commit": dict(getattr(self, "_room_last_commit", None) or {}),
            "graph_payload": graph_payload,
            "raw_occupancy_grid": self.latest_occupancy_grid,
        }

    def _set_planning_interaction_pending(self, node_id, pending):
        overlay_lock = getattr(self, "_planning_overlay_lock", None)
        if overlay_lock is None:
            return self.semantic_occ_overlay.set_interaction_pending(node_id, pending)
        with overlay_lock:
            return self.semantic_occ_overlay.set_interaction_pending(node_id, pending)

    def _reset_planning_overlay_state(self):
        overlay_lock = getattr(self, "_planning_overlay_lock", None)
        if overlay_lock is None:
            self.semantic_occ_overlay.reset()
            self.semantic_occ_update_tracker.reset()
            return
        with overlay_lock:
            self.semantic_occ_overlay.reset()
            self.semantic_occ_update_tracker.reset()
            self._planning_clear_mask_initialized = False
            self._planning_clear_mask_geometry_key = None
            self._planning_overlay_was_active = False

    def _build_grid_from_snapshot(self, data, info):
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.world_frame
        grid.info = info
        grid.data = [int(value) for value in data]
        return grid

    def _build_planning_products_from_snapshot(self, raw, graph_payload):
        """Materialize planning products without holding ``self.lock``.

        For the overwhelmingly common no-open-portal state, planning consumes
        the raw map directly.  A full all-zero door mask is only published at
        initialization, a geometry change, or an active-to-inactive overlay
        transition, so 5 Hz publishing does not repeatedly copy a 4M-cell mask.
        """

        empty_stats = {
            "active_portal_ids": [],
            "cleared_cells": 0,
            "update_bounds": None,
            "valid": False,
        }
        if raw is None:
            return None, None, None, empty_stats

        overlay_lock = getattr(self, "_planning_overlay_lock", None)
        if overlay_lock is None:
            # Compatibility path for minimal unit-test node doubles.
            return self._build_planning_products_locked(raw, graph_payload)
        with overlay_lock:
            self.semantic_occ_overlay.update_graph(graph_payload)
            geometry_key = self._occupancy_geometry_key(raw)
            overlay_active = bool(
                self.semantic_occ_overlay.enabled
                and self.semantic_occ_overlay.active_portal_ids
            )
            if not overlay_active:
                must_publish_zero_mask = (
                    not self._planning_clear_mask_initialized
                    or self._planning_overlay_was_active
                    or geometry_key != self._planning_clear_mask_geometry_key
                )
                door_clear_mask = None
                if must_publish_zero_mask:
                    zero_mask = [0] * (int(raw.info.width) * int(raw.info.height))
                    door_clear_mask = self._build_occupancy_copy(zero_mask, raw=raw)
                    self._planning_clear_mask_initialized = True
                    self._planning_clear_mask_geometry_key = geometry_key
                if self._planning_overlay_was_active:
                    # A full raw planning map is sent below, so incremental
                    # overlay state is no longer useful after the clear.
                    self.semantic_occ_update_tracker.reset()
                self._planning_overlay_was_active = False
                return raw, None, door_clear_mask, {
                    "active_portal_ids": [],
                    "cleared_cells": 0,
                    "update_bounds": None,
                    "valid": True,
                }

            planning_data, mask_data, overlay_stats = self.semantic_occ_overlay.apply(
                raw.info,
                raw.data,
                include_pending=True,
            )
            planning_grid = self._build_occupancy_copy(planning_data, raw=raw)
            door_clear_mask = self._build_occupancy_copy(mask_data, raw=raw)
            update_region = self.semantic_occ_update_tracker.build(
                raw.info.width,
                raw.info.height,
                planning_data,
                overlay_stats.get("update_bounds"),
                geometry_key=geometry_key,
            )
            planning_update = (
                self._build_occupancy_update(planning_grid, update_region)
                if update_region is not None
                else None
            )
            self._planning_clear_mask_initialized = True
            self._planning_clear_mask_geometry_key = geometry_key
            self._planning_overlay_was_active = True
            return planning_grid, planning_update, door_clear_mask, overlay_stats

    def _build_planning_products_locked(self, raw, graph_payload):
        """Legacy synchronous implementation for tests that bypass ``__init__``."""

        self.semantic_occ_overlay.update_graph(graph_payload)
        planning_data, mask_data, overlay_stats = self.semantic_occ_overlay.apply(
            raw.info,
            raw.data,
            include_pending=True,
        )
        planning_grid = self._build_occupancy_copy(planning_data, raw=raw)
        door_clear_mask = self._build_occupancy_copy(mask_data, raw=raw)
        update_region = self.semantic_occ_update_tracker.build(
            raw.info.width,
            raw.info.height,
            planning_data,
            overlay_stats.get("update_bounds"),
            geometry_key=self._occupancy_geometry_key(raw),
        )
        planning_update = (
            self._build_occupancy_update(planning_grid, update_region)
            if update_region is not None
            else None
        )
        return planning_grid, planning_update, door_clear_mask, overlay_stats

    def _build_publish_bundle_from_snapshot(self, snapshot):
        scene_info = snapshot.get("scene_info")
        scene_data = snapshot.get("scene_data")
        scene_confidence_data = snapshot.get("scene_confidence_data")
        scene_grid = (
            self._build_grid_from_snapshot(scene_data, scene_info)
            if (
                snapshot.get("publish_scene_grids")
                and scene_info is not None
                and scene_data is not None
                and scene_confidence_data is not None
            )
            else None
        )
        scene_conf_grid = (
            self._build_grid_from_snapshot(scene_confidence_data, scene_info)
            if scene_grid is not None
            else None
        )
        graph_payload = snapshot.get("graph_payload") or {}
        planning_grid, planning_update, door_clear_mask, overlay_stats = (
            self._build_planning_products_from_snapshot(
                snapshot.get("raw_occupancy_grid"),
                graph_payload,
            )
        )
        return {
            "obj_map": snapshot.get("obj_map"),
            "scene_grid": scene_grid,
            "scene_conf_grid": scene_conf_grid,
            "scene_revision": snapshot.get("scene_revision"),
            "room_segment_grid": snapshot.get("room_segment_grid"),
            "room_commit": snapshot.get("room_commit") or {},
            "graph_payload": graph_payload,
            # Keep the exact raw source in the emitted bundle.  Strict
            # readiness is evaluated after publishing and must never read a
            # newer ``latest_occupancy_grid`` by accident.
            "raw_occupancy_grid": snapshot.get("raw_occupancy_grid"),
            "room_attribute_request": self._build_room_attribute_request_locked(graph_payload),
            "planning_grid": planning_grid,
            "planning_update": planning_update,
            "door_clear_mask": door_clear_mask,
            "overlay_stats": overlay_stats,
        }

    def _collect_publish_bundle(self):
        """Snapshot mapper state briefly, then build all ROS products lock-free."""
        legacy_collector = self.__dict__.get("_collect_publish_bundle_locked")
        if legacy_collector is not None and not hasattr(self, "enable_scene_mapping"):
            return legacy_collector()
        with self.lock:
            snapshot = self._snapshot_publish_inputs_locked()
        build_t0 = time.perf_counter()
        bundle = self._build_publish_bundle_from_snapshot(snapshot)
        self._record_component_timing(
            "publish_bundle_build",
            (time.perf_counter() - build_t0) * 1000.0,
        )
        return bundle

    def _collect_publish_bundle_locked(self):
        """Compatibility wrapper for minimal synchronous test doubles.

        Production callbacks use ``_collect_publish_bundle`` so the expensive
        full-grid materialization is outside ``self.lock``.  Keep this name for
        older focused tests that call the legacy method directly.
        """
        snapshot = self._snapshot_publish_inputs_locked()
        return self._build_publish_bundle_from_snapshot(snapshot)

    def _publish_bundle(self, bundle):
        obj_map = bundle["obj_map"]
        scene_grid = bundle["scene_grid"]
        scene_conf_grid = bundle["scene_conf_grid"]
        scene_revision = bundle.get("scene_revision")
        room_segment_grid = bundle["room_segment_grid"]
        graph_payload = bundle["graph_payload"]
        room_attribute_request = bundle["room_attribute_request"]
        planning_grid = bundle["planning_grid"]
        planning_update = bundle["planning_update"]
        door_clear_mask = bundle["door_clear_mask"]
        if obj_map is not None:
            self.object_pub.publish(String(data=dumps_compact(obj_map)))
            self.marker_pub.publish(self._build_object_markers(obj_map))
        if scene_grid is not None and scene_conf_grid is not None:
            self.scene_id_pub.publish(scene_grid)
            self.scene_conf_pub.publish(scene_conf_grid)
            if scene_revision is not None:
                with self.lock:
                    if int(getattr(self, "_scene_grid_revision", 0)) == int(scene_revision):
                        self._scene_grid_published_revision = int(scene_revision)
        if room_segment_grid is not None:
            self.room_segment_pub.publish(room_segment_grid)
        if planning_grid is not None:
            self.planning_occupancy_grid_pub.publish(planning_grid)
        if door_clear_mask is not None:
            self.door_clear_mask_pub.publish(door_clear_mask)
        if planning_update is not None:
            self.planning_occupancy_grid_updates_pub.publish(planning_update)
        self.unified_graph_pub.publish(String(data=dumps_compact(graph_payload)))
        if room_attribute_request is not None:
            self.room_attribute_requests_pub.publish(
                String(data=dumps_compact(room_attribute_request))
            )
        self.navigation_hints_pub.publish(String(data=dumps_compact(graph_payload["views"]["navigation_view"]["hints"])))
        self.unified_graph_markers_pub.publish(build_graph_marker_array(graph_payload, self.world_frame))
        self.unified_graph_markers_lifted_pub.publish(
            build_graph_marker_array(
                graph_payload,
                self.lifted_graph_frame,
            )
        )
        self._save_graph_payload(graph_payload)

    def _build_occupancy_copy(self, data, *, raw=None):
        if raw is None:
            raw = self.latest_occupancy_grid
        grid = OccupancyGrid()
        # Keep the raw map timestamp so downstream consumers can pair the
        # semantic overlay and clear mask with the exact source occupancy map.
        grid.header.seq = raw.header.seq
        grid.header.stamp = raw.header.stamp
        grid.header.frame_id = raw.header.frame_id or self.world_frame
        grid.info = raw.info
        grid.data = [int(value) for value in data]
        return grid

    def _build_cropped_room_segment_grid(self, room_ids, *, raw=None):
        if raw is None:
            raw = self.latest_occupancy_grid
        width = int(raw.info.width)
        height = int(raw.info.height)
        values = room_ids
        valid_indices = [index for index, room_id in enumerate(values) if int(room_id) >= 0]

        if valid_indices:
            rows = [index // width for index in valid_indices]
            cols = [index % width for index in valid_indices]
            row_min = min(rows)
            row_max = max(rows) + 1
            col_min = min(cols)
            col_max = max(cols) + 1
        else:
            row_min = 0
            row_max = 1
            col_min = 0
            col_max = 1

        cropped = []
        for row in range(row_min, row_max):
            start = row * width + col_min
            cropped.extend(values[start : start + (col_max - col_min)])

        grid = OccupancyGrid()
        grid.header.seq = raw.header.seq
        grid.header.stamp = raw.header.stamp
        grid.header.frame_id = raw.header.frame_id or self.world_frame
        grid.info = copy.deepcopy(raw.info)
        grid.info.width = col_max - col_min
        grid.info.height = row_max - row_min

        origin = grid.info.origin
        yaw = math.atan2(
            2.0 * (origin.orientation.w * origin.orientation.z + origin.orientation.x * origin.orientation.y),
            1.0 - 2.0 * (origin.orientation.y * origin.orientation.y + origin.orientation.z * origin.orientation.z),
        )
        resolution = float(grid.info.resolution)
        offset_x = float(col_min) * resolution
        offset_y = float(row_min) * resolution
        origin.position.x += math.cos(yaw) * offset_x - math.sin(yaw) * offset_y
        origin.position.y += math.sin(yaw) * offset_x + math.cos(yaw) * offset_y
        grid.data = [int(value) for value in cropped]
        return grid

    @staticmethod
    def _occupancy_geometry_key(grid):
        info = grid.info
        origin = info.origin
        return (
            int(info.width),
            int(info.height),
            round(float(info.resolution), 9),
            round(float(origin.position.x), 6),
            round(float(origin.position.y), 6),
            round(float(origin.orientation.z), 6),
            round(float(origin.orientation.w), 6),
            str(grid.header.frame_id),
        )

    @staticmethod
    def _build_occupancy_update(planning_grid, region):
        update = OccupancyGridUpdate()
        update.header = planning_grid.header
        update.x = int(region["x"])
        update.y = int(region["y"])
        update.width = int(region["width"])
        update.height = int(region["height"])
        update.data = [int(value) for value in region["data"]]
        return update

    def _stamp_from_detection_payload(self, parsed):
        secs = parsed.get("secs")
        nsecs = parsed.get("nsecs")
        if secs is None:
            return rospy.Time.now().to_sec()
        try:
            return float(secs) + float(nsecs or 0) * 1e-9
        except (TypeError, ValueError):
            return rospy.Time.now().to_sec()

    @staticmethod
    def _capture_step_from_payload(parsed):
        if not isinstance(parsed, dict):
            return None
        value = parsed.get("capture_step")
        if value is None:
            value = parsed.get("frame_index")
        try:
            capture_step = int(value)
        except (TypeError, ValueError):
            return None
        return capture_step if capture_step >= 0 else None

    def _publish_lifted_graph_tf(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = self.lifted_graph_frame
        tf_msg.transform.translation.z = float(self.lifted_graph_z_offset)
        tf_msg.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(tf_msg)

    def _segment_rooms_from_occupancy(self, occ_grid, *, force_stable=False):
        return self.room_segmenter.segment(occ_grid, force_stable=force_stable)

    def _save_graph_payload(self, graph_payload):
        if not self.graph_save_path:
            return
        target_dir = os.path.dirname(self.graph_save_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        with open(self.graph_save_path, "w", encoding="utf-8") as handle:
            if self.graph_save_pretty:
                json.dump(graph_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            else:
                json.dump(graph_payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def save_graph_callback(self, _request):
        with self.lock:
            path = self._save_episode_graph_locked(final=False)
        if path is None:
            return TriggerResponse(success=False, message="semantic_map/graph_save_dir is empty")
        return TriggerResponse(success=True, message=path)

    def _save_episode_graph_locked(self, final=False):
        if not self.graph_save_dir or not self.graph_store.nodes:
            return None
        graph_payload = self.graph_store.as_graph_dict()
        episode_id = graph_payload.get("episode_id") or "episode_unknown"
        revision = int(graph_payload.get("graph_revision", 0))
        suffix = "final" if final else f"revision_{revision}"
        target_path = os.path.join(self.graph_save_dir, f"{episode_id}_{suffix}.json")
        os.makedirs(self.graph_save_dir, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as handle:
            if self.graph_save_pretty:
                json.dump(graph_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            else:
                json.dump(graph_payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return target_path

    def _build_object_markers(self, obj_map):
        markers = MarkerArray()
        now = rospy.Time.now()
        for idx, obj in enumerate(obj_map):
            coord = obj.get("coord", [0.0, 0.0, 0.0])
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "semantic_objects_py"
            marker.id = idx * 2
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(coord[0])
            marker.pose.position.y = float(coord[1])
            marker.pose.position.z = float(coord[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.25
            marker.color.r = 0.1
            marker.color.g = 0.7
            marker.color.b = 1.0
            marker.color.a = max(0.2, min(1.0, float(obj.get("conf", 0.5))))
            markers.markers.append(marker)

            text = Marker()
            text.header = marker.header
            text.ns = "semantic_object_labels_py"
            text.id = idx * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(coord[0])
            text.pose.position.y = float(coord[1])
            text.pose.position.z = float(coord[2]) + 0.35
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = str(obj.get("semantic_name", "object"))
            markers.markers.append(text)
        return markers


if __name__ == "__main__":
    SemanticMappingNode()
    rospy.spin()
