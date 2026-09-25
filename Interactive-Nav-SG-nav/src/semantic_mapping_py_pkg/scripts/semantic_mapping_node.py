#!/usr/bin/env python3
import copy
import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from contextlib import nullcontext

import numpy as np
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
from semantic_mapping_py_pkg.messages import dumps_compact, parse_json_list, parse_json_object_or_text, observation_stamp_seconds
from semantic_mapping_py_pkg.occupancy_transport import NumpyOccupancyGrid, occupancy_data_snapshot
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter, RoomSegmentationState
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics
from semantic_mapping_py_pkg.room_inference_backends import normalized_room_box, room_evidence_signature
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore, SceneGridStore
from semantic_mapping_py_pkg.semantic_occ_overlay import OverlayUpdateRegionTracker, SemanticOccupancyOverlay
from semantic_mllm_py_pkg.ablation import AblationConfig


class SemanticMappingNode:
    def __init__(self):
        patch_roslogging_findcaller_for_py311()
        rospy.init_node("semantic_mapping_py")
        # The tracked-detection stream is also the cache epoch for the
        # asynchronous Module-1 worker.  A mapper restart clears the graph,
        # so publish a fresh epoch even when the physical detector keeps the
        # same track IDs; otherwise M1 can retain a completed locker
        # recheck and never repopulate the newly-created graph node.
        self.tracked_stream_epoch = f"mapper_{os.getpid()}_{time.time_ns()}"
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
        self.base_frame = frames.get("base_frame", "tf_frame_base_link")
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
        self.tracked_detections_topic = topics.get(
            "tracked_detections", "/semantic_mapping/tracked_detections"
        )
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
        # Scene attributes are projected through ``latest_cloud``.  The
        # realtime-GT path never publishes those legacy attributes, so its
        # parent launch may opt out of the otherwise expensive PointCloud2
        # subscription/deserialization.  Keep the default compatible with the
        # detector and offline scene-mapping paths.
        self.subscribe_pointcloud = bool(config.get("subscribe_pointcloud", True))
        self.publish_rate = float(config.get("publish_rate", 2.0))
        # Full graph/planning products are large ROS messages.  The timer may
        # still run at the requested UI rate, but an unchanged input bundle
        # only needs a low-rate heartbeat for late subscribers.
        self.publish_heartbeat_period_sec = max(
            0.0, float(config.get("publish_heartbeat_period_sec", 1.0))
        )
        # Graph JSON/markers are latched state products; rebuilding them for
        # every raw OCC frame burns CPU because visibility updates can bump
        # the internal revision even when topology is unchanged. Keep a
        # bounded heartbeat so subscribers receive fresh state without
        # coupling OCC cadence to graph serialization.
        self.graph_publish_period_sec = max(
            0.05, float(config.get("graph_publish_period_sec", 0.2))
        )
        # Room topology changes much more slowly than detector/object state.
        # Keep the mapper timer responsive while rate-limiting the large
        # room-segment OccupancyGrid independently.
        self.room_grid_publish_rate = max(
            0.1, float(config.get("room_grid_publish_rate", 2.0))
        )
        self.room_segment_min_interval_sec = max(
            0.0, float(config.get("room_segment_min_interval_sec", 0.5))
        )
        self._room_last_worker_start_mono = 0.0
        self._last_room_grid_publish_mono = 0.0
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
        self.room_enclosed_occupied_max_aspect = float(config.get("room_enclosed_occupied_max_aspect", 1.8))
        self.room_enclosed_occupied_known_ring_ratio = float(
            config.get("room_enclosed_occupied_known_ring_ratio", 0.95)
        )
        self.room_enclosed_occupied_free_ring_ratio = float(
            config.get("room_enclosed_occupied_free_ring_ratio", 0.45)
        )
        self.room_fill_enclosed_unknown = bool(
            config.get("room_fill_enclosed_unknown", False)
        )
        self.room_enclosed_unknown_max_cells = max(
            0, int(config.get("room_enclosed_unknown_max_cells", 250))
        )
        self.room_enclosed_unknown_known_ring_ratio = float(
            config.get("room_enclosed_unknown_known_ring_ratio", 0.90)
        )
        self.room_enclosed_unknown_free_ring_ratio = float(
            config.get("room_enclosed_unknown_free_ring_ratio", 0.75)
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
        # Exact topology reuse is deliberately conservative: it only applies
        # after the RoomSegmenter temporal state has settled and the canonical
        # free/occupied/unknown grid is byte-for-byte unchanged.
        self.room_topology_cache_enabled = bool(
            config.get("room_topology_cache_enabled", True)
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
        self.room_mllm_success_refresh_interval_s = max(
            0.0, float(room_mllm_config.get("success_refresh_interval_s", 120.0))
        )
        # Room MLLM evidence is intentionally incremental.  A room remains
        # dirty until a non-fallback result for the same evidence signature is
        # applied, so a failed request can be refreshed without re-submitting
        # every room in the graph.
        self._room_mllm_committed_signatures = {}
        self._room_mllm_committed_at = {}
        self._room_mllm_episode_id = ""
        self._room_mllm_selection_cursor = 0
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
        # Newly exposed refrigerator contents are often visible for fewer
        # frames than the strict interaction-object confirmation window.  Keep
        # the M1/public tracked-detection stream strict, but optionally admit
        # currently visible tentative tracks to the graph while a trusted
        # refrigerator-open state is active.  The graph's geometry/provenance
        # filters remain the final containment gate.
        self.open_refrigerator_content_mapping_enabled = bool(
            config.get("open_refrigerator_content_mapping_enabled", False)
        )
        self.open_refrigerator_content_min_observations = max(
            1,
            int(config.get("open_refrigerator_content_min_observations", 1)),
        )
        self.open_refrigerator_content_ignore_class_confirmations = bool(
            config.get(
                "open_refrigerator_content_ignore_class_confirmations", True
            )
        )
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
            duplicate_bbox_iou_threshold=config.get("object_duplicate_bbox_iou_threshold", 0.0),
            duplicate_3d_overlap_threshold=config.get("object_duplicate_3d_overlap_threshold", 0.15),
            class_min_confirmations=config.get("object_class_min_confirmations", {}),
            class_min_top_height_m=config.get("object_class_min_top_height_m", {}),
            portal_cross_view_match_enabled=config.get(
                "object_portal_cross_view_match_enabled", False
            ),
            portal_cross_view_normal_distance_m=config.get(
                "object_portal_cross_view_normal_distance_m", 0.45
            ),
            portal_cross_view_yaw_tolerance_rad=config.get(
                "object_portal_cross_view_yaw_tolerance_rad", 0.35
            ),
            portal_cross_view_min_tangent_overlap_ratio=config.get(
                "object_portal_cross_view_min_tangent_overlap_ratio", 0.20
            ),
            portal_cross_view_min_vertical_overlap_ratio=config.get(
                "object_portal_cross_view_min_vertical_overlap_ratio", 0.35
            ),
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
            clear_padding_m=overlay_config.get("clear_padding_m", -0.05),
            open_states=overlay_config.get("open_states", ["open"]),
            max_aperture_thickness_m=overlay_config.get(
                "max_aperture_thickness_m", 0.35
            ),
            raw_free_confirmations=overlay_config.get(
                "raw_free_confirmations", 3
            ),
        )
        # Room segmentation consumes the same graph snapshot as the planning
        # overlay, but it must not share the mutable planning-overlay instance:
        # the room worker runs outside ``self.lock`` while the publishing path
        # continues to update the planning overlay under that lock.
        self._room_segmentation_overlay = SemanticOccupancyOverlay(
            enabled=overlay_config.get("enabled", True),
            clear_padding_m=overlay_config.get("clear_padding_m", -0.05),
            open_states=overlay_config.get("open_states", ["open"]),
            max_aperture_thickness_m=overlay_config.get(
                "max_aperture_thickness_m", 0.35
            ),
            raw_free_confirmations=overlay_config.get(
                "raw_free_confirmations", 3
            ),
        )
        self.semantic_occ_update_tracker = OverlayUpdateRegionTracker()
        # Planning overlay state is shared by interaction callbacks and the
        # timer/graph publishers.  It has its own lock so the expensive map
        # materialization never requires the mapper data lock.
        self._planning_overlay_lock = threading.RLock()
        self._planning_clear_mask_initialized = False
        self._planning_clear_mask_geometry_key = None
        self._planning_overlay_was_active = False
        # Materialized planning products are keyed by the *content* of the
        # source occupancy grid, its geometry, and the effective portal
        # overlay.  Header/sequence changes alone must only retime a shallow
        # ROS message, not copy a multi-million-cell list again.
        self._planning_products_cache = None

        self.lock = threading.Lock()
        # ``room_segmenter`` has temporal state (stable IDs, portal hints and
        # merge confirmation).  Keep that state serial, but deliberately keep
        # it separate from the mapper's short-lived data/graph lock.
        self._room_lock = threading.RLock()
        # ROS callbacks and the room worker may both request a bundle.  One
        # re-entrant owner keeps a causal commit bundle contiguous through its
        # ready watermark while preserving the legacy direct callback paths.
        self._publish_lock = threading.RLock()
        self._room_work_condition = threading.Condition()
        self._room_work_pending = None
        self._room_portal_pending = deque(maxlen=64)
        self._room_portal_dropped_batches = 0
        self._room_state_epoch = 0
        self._room_worker_stopping = False
        self._room_worker_thread = None
        # Detector receipts are latest-only as well.  A slow graph update must
        # never let a queue of old YOLO JSON messages build up and replay stale
        # boxes after the robot has moved.
        self._object_work_condition = threading.Condition()
        self._object_work_pending = None
        self._object_worker_stopping = False
        self._object_worker_thread = None
        self._room_input_revision = 0
        # Raw OCC updates do not invalidate an in-flight room segmentation when
        # the grid geometry is unchanged.  Portal/episode structure changes do.
        self._room_topology_revision = 0
        self._room_epoch = 0
        self._room_last_committed_revision = -1
        self._room_topology_cache = None
        # Materialized room-overlay occupancy is keyed by raw cell content
        # and the confirmed portal geometry.  Room refresh requests are
        # latest-only, but GMapping commonly republishes an unchanged map
        # with a newer header; retaining this list avoids copying millions of
        # cells through ``SemanticOccupancyOverlay.apply`` on every request.
        self._room_overlay_cache = None
        # ``_room_topology_cache_key`` used to allocate an int16 and uint8
        # array plus a BLAKE2 digest on every OCC receipt, even when GMapping
        # had published the exact same cells with only a newer header.  Keep a
        # private copy of the last raw values and their ternary categories so
        # unchanged frames can take the list-equality fast path.  The copy is
        # deliberate: a few in-process producers reuse and mutate their ROS
        # message buffer in place, so trusting ``data is previous`` would make
        # a stale topology cache possible.
        self._room_topology_input_cache = None
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
        self._last_publish_signature = None
        self._last_publish_mono = 0.0
        self._graph_payload_cache_token = None
        self._graph_payload_cache = None
        self._last_graph_publish_token = None
        self._last_graph_publish_mono = 0.0
        # A successful portal opening is followed by one immediate planning
        # map publication from the first newer raw OCC.  The normal timer is
        # intentionally not accelerated for every SLAM frame: this narrow
        # hand-off gives the post-open causal gate a source-matched map
        # without turning full-map publishing into the hot path.
        self._post_open_planning_refresh_after_stamp_sec = None
        self._post_open_room_refresh_result = None
        self.pending_interaction_commands = {}
        self.max_pending_interaction_commands = 128
        self.require_interaction_command_id = bool(config.get("require_interaction_command_id", False))
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
            room_fill_enclosed_unknown=self.room_fill_enclosed_unknown,
            room_enclosed_unknown_max_cells=self.room_enclosed_unknown_max_cells,
            room_enclosed_unknown_known_ring_ratio=self.room_enclosed_unknown_known_ring_ratio,
            room_enclosed_unknown_free_ring_ratio=self.room_enclosed_unknown_free_ring_ratio,
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

        # Create every output publisher before registering input subscribers.
        # rospy may deliver a latched/high-rate detection immediately during
        # Subscriber construction; registering the subscribers first allowed
        # ``object_callback`` to build a graph and then crash while publishing
        # because (for example) ``planning_occupancy_grid_pub`` did not exist
        # yet.  That startup race dropped the first refrigerator observations
        # and could leave the graph without the appliance/content relation.
        self.step_ready_pub = rospy.Publisher("/semantic_decision/ready/semantic_mapping", String, queue_size=32)
        self.object_pub = rospy.Publisher(self.object_map_topic, String, queue_size=1)
        self.tracked_detections_pub = rospy.Publisher(
            self.tracked_detections_topic, String, queue_size=1
        )
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

        self.object_sub = rospy.Subscriber(self.object_detection_topic, String, self.object_callback, queue_size=1)
        self.scene_sub = rospy.Subscriber(self.scene_attribute_topic, String, self.scene_callback, queue_size=10)
        self.cloud_sub = None
        if self.subscribe_pointcloud:
            self.cloud_sub = rospy.Subscriber(
                self.pointcloud_topic,
                PointCloud2,
                self.pointcloud_callback,
                queue_size=1,
            )
        else:
            rospy.loginfo(
                "[semantic_mapping_node.py] PointCloud2 subscription disabled; "
                "legacy scene_attribute messages will be ignored"
            )
        self.occ_sub = rospy.Subscriber(self.occupancy_grid_topic, NumpyOccupancyGrid, self.occupancy_callback, queue_size=1)
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
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_callback)
        self.save_graph_service = rospy.Service("/semantic_mapping/save_graph", Trigger, self.save_graph_callback)
        self._publish_lifted_graph_tf()
        self._start_room_worker()
        self._start_object_worker()

        rospy.loginfo(
            "[semantic_mapping_node.py] object_in=%s scene_in=%s cloud=%s occ=%s",
            self.object_detection_topic,
            self.scene_attribute_topic,
            self.pointcloud_topic if self.subscribe_pointcloud else "<disabled>",
            self.occupancy_grid_topic,
        )

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

    def _start_object_worker(self):
        """Start the latest-only detector/graph worker."""

        worker = threading.Thread(
            target=self._object_worker_loop,
            name="semantic_object_mapping",
            daemon=True,
        )
        self._object_worker_thread = worker
        worker.start()
        try:
            rospy.on_shutdown(self._stop_object_worker)
        except AttributeError:
            pass

    def _stop_object_worker(self):
        condition = getattr(self, "_object_work_condition", None)
        if condition is None:
            return
        with condition:
            self._object_worker_stopping = True
            condition.notify_all()

    def _object_worker_loop(self):
        while True:
            with self._object_work_condition:
                while (
                    self._object_work_pending is None
                    and not self._object_worker_stopping
                ):
                    self._object_work_condition.wait()
                if self._object_worker_stopping:
                    return
                payload = self._object_work_pending
                self._object_work_pending = None
            try:
                self._process_object_message(payload)
            except Exception as exc:
                if not rospy.is_shutdown():
                    rospy.logerr(
                        "[semantic_mapping_node.py] object worker failed: %s",
                        exc,
                    )

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

    def _reset_room_worker_state(self, *, worker_owned=False):
        """Reset worker-owned temporal state outside the mapper data lock."""

        if not worker_owned and getattr(self, "_room_worker_thread", None) is not None:
            # The worker clears its state before processing the new epoch.
            # A ROS reset callback must not wait for an old segmentation.
            self._enqueue_room_refresh(reason="episode_reset")
            return
        room_lock = getattr(self, "_room_lock", None)
        if room_lock is None:
            self.room_segmenter.state = RoomSegmentationState()
            overlay = getattr(self, "_room_segmentation_overlay", None)
            if overlay is not None:
                overlay.reset()
            self._room_topology_cache = None
            self._room_topology_input_cache = None
            self._room_overlay_cache = None
            return
        with room_lock:
            self.room_segmenter.state = RoomSegmentationState()
            self._room_segmentation_overlay.reset()
            self._room_topology_cache = None
            self._room_topology_input_cache = None
            self._room_overlay_cache = None

    def _update_room_portal_hints(self, observations, *, source_mode, refresh_active=False, epoch=None):
        """Queue portal evidence without waiting for background segmentation."""

        condition = getattr(self, "_room_work_condition", None)
        if condition is not None and getattr(self, "_room_worker_thread", None) is not None:
            observations = [item for item in observations or []
                            if RoomSegmenter._is_portal_observation(item)]
            if not observations:
                return False
            with condition:
                if self._room_worker_stopping:
                    return False
                current_epoch = int(getattr(self, "_room_epoch", 0))
                epoch = current_epoch if epoch is None else int(epoch)
                if epoch != current_epoch:
                    return False
                pending = self._room_portal_pending
                if len(pending) == pending.maxlen:
                    self._room_portal_dropped_batches += 1
                pending.append((epoch, list(observations), source_mode, refresh_active))
            self._enqueue_room_refresh(reason="portal_hint_pending", epoch=epoch)
            return False  # Structural revision changes when the worker applies it.
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

    def _consume_room_portal_hints(self, epoch):
        """Apply bounded observation batches only on the room worker."""
        condition = getattr(self, "_room_work_condition", None)
        if condition is None or not hasattr(self, "_room_portal_pending"):
            return
        with condition:
            if int(epoch) != int(self._room_epoch):
                return
            batches = list(self._room_portal_pending)
            self._room_portal_pending.clear()
        changed = False
        with self._room_lock:
            if int(epoch) != int(self._room_epoch):
                return
            if int(self._room_state_epoch) != int(epoch):
                self._reset_room_worker_state(worker_owned=True)
                self._room_state_epoch = int(epoch)
            for batch_epoch, observations, source_mode, refresh_active in batches:
                if batch_epoch == int(epoch):
                    changed = self.room_segmenter.update_portal_hints(
                        observations, source_mode=source_mode,
                        refresh_active=refresh_active,
                    ) or changed
        if changed:
            with self.lock:
                if int(epoch) == int(self._room_epoch):
                    self._mark_room_inputs_dirty_locked(structural=True)

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
        epoch=None,
        urgent=False,
    ):
        """Coalesce a room refresh request without retaining stale grids.

        The compatibility fallback is useful for the small node doubles used by
        unit tests and for callers that construct a node without ``__init__``.
        Production nodes always use the background worker.
        """

        condition = getattr(self, "_room_work_condition", None)
        worker = getattr(self, "_room_worker_thread", None)
        if condition is None or worker is None:
            if epoch is not None and int(epoch) != int(getattr(self, "_room_epoch", 0)):
                return False
            if post_open_result is not None:
                return self._refresh_confirmed_open_portal_room_grid_locked(post_open_result)
            self._run_sync_room_refresh_for_compat(force_stable=force_stable)
            return True

        with condition:
            if self._room_worker_stopping:
                return False
            with self.lock:
                if epoch is not None and int(epoch) != int(getattr(self, "_room_epoch", 0)):
                    return False
                if source is None:
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
            if pending is not None and int(pending.get("epoch", -1)) == current_epoch:
                post_open_results.extend(pending.get("post_open_results") or [])
                reasons.update(pending.get("reasons") or [])
                force_stable = bool(force_stable or pending.get("force_stable"))
                urgent = bool(urgent or pending.get("urgent"))
            if post_open_result is not None:
                post_open_results.append(dict(post_open_result))
            self._room_work_pending = {
                "input_revision": current_revision,
                "topology_revision": current_topology_revision,
                "epoch": current_epoch,
                "force_stable": bool(force_stable),
                "post_open_results": post_open_results,
                "reasons": reasons,
                # A content-superseded result must run immediately after the
                # current job returns.  The normal minimum interval is useful
                # for ordinary OCC heartbeats, but sleeping here would leave
                # a fresh topology behind another stale interval.
                "urgent": bool(urgent),
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
            if not bool(request.get("urgent", False)):
                wait_s = float(getattr(self, "room_segment_min_interval_sec", 0.0)) - (
                    time.monotonic() - float(
                        getattr(self, "_room_last_worker_start_mono", 0.0)
                    )
                )
                if wait_s > 0.0:
                    time.sleep(wait_s)
            self._room_last_worker_start_mono = time.monotonic()
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

    def _room_topology_cache_key(self, occupancy, *, topology_revision, epoch):
        """Return an exact, source-independent topology cache key.

        The room algorithm only distinguishes unknown, traversable and blocked
        cells.  A digest is used as a cheap prefilter, while the caller retains
        the canonical array and performs ``array_equal`` before reuse.  The raw
        source token is intentionally absent: equivalent map content may reuse
        computation, but each source still receives its own graph commit and
        causal ready message.
        """

        if occupancy is None:
            return None, None
        width = int(getattr(occupancy.info, "width", 0))
        height = int(getattr(occupancy.info, "height", 0))
        if width <= 0 or height <= 0:
            return None, None
        raw_data = getattr(occupancy, "data", ())
        expected_size = width * height
        input_cache = getattr(self, "_room_topology_input_cache", None)
        values = None
        categories = None
        digest = None
        if (
            isinstance(input_cache, dict)
            and input_cache.get("geometry") == self._occupancy_geometry_key(occupancy)
            and int(input_cache.get("size", -1)) == expected_size
            and int(input_cache.get("free_threshold", self.room_free_threshold))
            == int(self.room_free_threshold)
        ):
            # Compare against an owned list rather than the source object.  A
            # list comparison is implemented in C and avoids the NumPy
            # conversion/allocation on the common unchanged-map path.
            try:
                comparable_raw_data = (
                    raw_data if isinstance(raw_data, list) else list(raw_data)
                )
                if comparable_raw_data == input_cache.get("raw_data"):
                    categories = input_cache.get("categories")
                    digest = input_cache.get("digest")
                # ``raw_data`` may be a tuple/array in tests or custom bridge
                # code; the comparison above remains correct, while malformed
                # values fall through to the existing validation path.
            except (TypeError, ValueError):
                categories = None
        if categories is None:
            values = np.asarray(raw_data, dtype=np.int16)
            if values.size != expected_size:
                return None, None
            categories = np.empty(values.shape, dtype=np.uint8)
            categories[values < 0] = 0
            categories[(values >= 0) & (values <= self.room_free_threshold)] = 1
            categories[values > self.room_free_threshold] = 2
            digest = hashlib.blake2b(
                memoryview(categories), digest_size=16
            ).digest()
            try:
                owned_raw_data = list(raw_data)
            except (TypeError, ValueError):
                owned_raw_data = None
            self._room_topology_input_cache = {
                "geometry": self._occupancy_geometry_key(occupancy),
                "size": expected_size,
                "free_threshold": int(self.room_free_threshold),
                "raw_data": owned_raw_data,
                "categories": categories,
                "digest": digest,
            }
        geometry = self._occupancy_geometry_key(occupancy)
        segmenter = self.room_segmenter
        config_signature = (
            segmenter.room_free_threshold,
            segmenter.room_unknown_id,
            segmenter.room_min_component_cells,
            segmenter.room_core_min_component_cells,
            segmenter.room_core_clearance_cells,
            segmenter.room_small_obstacle_max_cells,
            segmenter.room_remove_enclosed_occupied,
            segmenter.room_enclosed_occupied_max_cells,
            segmenter.room_enclosed_occupied_max_aspect,
            segmenter.room_enclosed_occupied_known_ring_ratio,
            segmenter.room_enclosed_occupied_free_ring_ratio,
            segmenter.room_fill_enclosed_unknown,
            segmenter.room_enclosed_unknown_max_cells,
            segmenter.room_enclosed_unknown_known_ring_ratio,
            segmenter.room_enclosed_unknown_free_ring_ratio,
            segmenter.room_fill_enclosed_obstacles,
            segmenter.room_enclosed_obstacle_min_cells,
            segmenter.room_enclosed_obstacle_max_cells,
            segmenter.room_enclosed_obstacle_dominance_ratio,
            segmenter.room_portal_cut_enabled,
            segmenter.room_portal_cut_margin_m,
            segmenter.room_portal_cut_thickness_cells,
            segmenter.room_id_overlap_ratio,
            segmenter.room_merge_confirmations,
            segmenter.room_grid_stability_frames,
        )
        return (
            int(epoch),
            int(topology_revision),
            geometry,
            config_signature,
            digest,
        ), categories

    def _room_occupancy_content_key(self, occupancy):
        """Return the room-segmentation input content key.

        Room segmentation deliberately classifies a map into only three
        states (unknown, free and occupied).  Header sequence/stamp changes,
        and changes to a free cell's numeric value within the configured free
        range, do not alter the segmentation result.  Keeping this key next to
        the worker snapshot lets a completed job be retimed to the newest ROS
        header without publishing a result for a genuinely different map.

        The digest is intentionally over the ternary mask rather than the raw
        int8 values.  It is a compact prefilter; geometry and cell count are
        included so maps with different layouts cannot alias in normal use.
        Malformed input returns ``None`` and therefore never takes the retime
        fast path.
        """

        if occupancy is None:
            return None
        try:
            info = occupancy.info
            width = int(info.width)
            height = int(info.height)
            expected = width * height
            if width <= 0 or height <= 0:
                return None
            values = np.asarray(getattr(occupancy, "data", ()), dtype=np.int16).reshape(-1)
            if int(values.size) != expected:
                return None
            threshold = int(getattr(self, "room_free_threshold", 20))
            categories = np.empty(values.shape, dtype=np.uint8)
            categories[values < 0] = 0
            categories[(values >= 0) & (values <= threshold)] = 1
            categories[values > threshold] = 2
            digest = hashlib.blake2b(
                memoryview(np.ascontiguousarray(categories)), digest_size=16
            ).digest()
            return (
                self._occupancy_geometry_key(occupancy),
                int(threshold),
                int(expected),
                digest,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

    def _resolve_room_refresh_commit_source(
        self,
        raw,
        room_grid,
        input_revision,
        content_key,
        *,
        epoch,
        topology_revision,
    ):
        """Resolve the newest source for a completed room segmentation.

        A worker may spend longer than one OCC period in ``segment``.  Before
        committing, compare the ternary content of the map used for that
        computation with the newest map.  Equivalent content is safe to reuse
        and is shallowly retimed; changed content is explicitly discarded so a
        stale room grid can never reach ROS or the strict ready barrier.

        The returned tuple is ``(raw, room_grid, input_revision, status)``.
        ``status`` is ``current``, ``retimed``, ``content_changed``,
        ``raced`` or ``stale_topology``.  The second lock check closes the
        small race between hashing the latest message and committing it.
        """

        with self.lock:
            if int(epoch) != int(getattr(self, "_room_epoch", 0)):
                return None, None, None, "stale_topology"
            if int(topology_revision) != int(
                getattr(self, "_room_topology_revision", 0)
            ):
                return None, None, None, "stale_topology"
            latest_raw = getattr(self, "latest_occupancy_grid", None)
            latest_revision = int(getattr(self, "_room_input_revision", 0))

        if latest_raw is raw and latest_revision == int(input_revision):
            return raw, room_grid, latest_revision, "current"

        latest_key = self._room_occupancy_content_key(latest_raw)
        if latest_key != content_key:
            return None, None, None, "content_changed"

        # Validate that the message hashed above is still the latest one.  If
        # another callback won the race, let the next latest-only request
        # retime/retry rather than publishing an intermediate source.
        with self.lock:
            if int(epoch) != int(getattr(self, "_room_epoch", 0)):
                return None, None, None, "stale_topology"
            if int(topology_revision) != int(
                getattr(self, "_room_topology_revision", 0)
            ):
                return None, None, None, "stale_topology"
            if (
                getattr(self, "latest_occupancy_grid", None) is not latest_raw
                or int(getattr(self, "_room_input_revision", 0)) != latest_revision
            ):
                return None, None, None, "raced"

        # ``room_grid.data`` is immutable after construction.  Reusing it
        # avoids a second full-map conversion while the new ROS header/info
        # establishes the causal source identity.
        # Even an in-place producer may advance the header on the same ROS
        # object.  Always rebuild the shallow shell when the revision moved;
        # only the immutable cell payload is shared.
        room_grid = self._retime_cached_grid(
            getattr(room_grid, "data", room_grid), latest_raw
        )
        return latest_raw, room_grid, latest_revision, "retimed"

    def _room_topology_cache_is_quiescent(self, *, force_stable):
        """Whether skipping ``segment`` cannot change its temporal state."""

        if force_stable or not getattr(self, "room_topology_cache_enabled", False):
            return False
        state = getattr(getattr(self, "room_segmenter", None), "state", None)
        if state is None:
            return False
        if getattr(state, "stable_room_ids", None) is None:
            return False
        if int(getattr(state, "candidate_room_count", 0)) < int(
            getattr(self, "room_grid_stability_frames", 1)
        ):
            return False
        return not (
            getattr(state, "pending_merges", None)
            or getattr(state, "last_confirmed_merges", None)
        )

    def _load_room_topology_cache(
        self, occupancy, *, topology_revision, epoch, force_stable
    ):
        """Return an exact cached room result or ``None``.

        This method is called under ``_room_lock`` and never advances
        RoomSegmenter state.  It therefore only admits the quiescent condition
        above; stability/merge transitions always take the full path.
        """

        if not self._room_topology_cache_is_quiescent(force_stable=force_stable):
            return None, None, None
        key, categories = self._room_topology_cache_key(
            occupancy,
            topology_revision=topology_revision,
            epoch=epoch,
        )
        entry = getattr(self, "_room_topology_cache", None)
        if key is None or not isinstance(entry, dict) or entry.get("key") != key:
            return None, key, categories
        # ``_room_topology_cache_key`` reuses the canonical category array
        # retained by ``_room_topology_input_cache`` when the raw OCC list is
        # unchanged.  The old unconditional ``np.array_equal`` scanned the
        # complete map a second time on every room request (often a million
        # cells) even though the digest/key comparison had already matched.
        # Retain the exact comparison as a fallback for test/custom callers
        # that provide an equivalent but separately allocated array.
        cached_categories = entry.get("categories")
        if cached_categories is not categories and not np.array_equal(
            cached_categories, categories
        ):
            return None, key, categories
        return entry, key, categories

    def _store_room_topology_cache(
        self,
        occupancy,
        *,
        topology_revision,
        epoch,
        force_stable,
        room_ids,
        room_conf,
        room_merges,
        cache_key=None,
        cache_categories=None,
    ):
        if room_merges or not self._room_topology_cache_is_quiescent(
            force_stable=force_stable
        ):
            self._room_topology_cache = None
            return
        key = cache_key
        categories = cache_categories
        if key is None or categories is None:
            key, categories = self._room_topology_cache_key(
                occupancy,
                topology_revision=topology_revision,
                epoch=epoch,
            )
        if key is None:
            self._room_topology_cache = None
            return
        self._room_topology_cache = {
            "key": key,
            "categories": categories,
            # Tuples make the reusable state immutable to graph/crop callers.
            # ``tuple(list)`` uses the C-level sequence copy; avoid a Python
            # generator over every map cell on every cache refresh.
            "room_ids": room_ids if isinstance(room_ids, tuple) else tuple(room_ids),
            "room_conf": room_conf if isinstance(room_conf, tuple) else tuple(room_conf),
        }

    def _process_room_refresh_request(self, request):
        """Build a room grid outside ``self.lock`` and commit it by revision."""

        total_t0 = time.perf_counter()
        self._consume_room_portal_hints(int(request.get("epoch", -1)))
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
            # Keep the graph-store snapshot under the mapper lock, but defer
            # Module-1's deepcopy until after releasing it.  Room segmentation
            # is intentionally asynchronous; a large graph must not block an
            # incoming OCC callback while the policy payload is copied.
            graph_payload_raw = self.graph_store.as_graph_dict()
            graph_module1_mode = self.ablation.module1
            post_open_references = []
            for result in request.get("post_open_results") or []:
                reference = self._confirmed_open_portal_reference_locked(result)
                if reference is not None:
                    post_open_references.append(reference)
        snapshot_ms = (time.perf_counter() - snapshot_t0) * 1000.0
        # Capture the exact semantic input used by this job.  RoomSegmenter
        # only distinguishes unknown/free/occupied cells, so a newer header or
        # a changed free-space confidence can be retimed later without rerun.
        segmentation_content_key = self._room_occupancy_content_key(raw)

        graph_ablation_t0 = time.perf_counter()
        graph_payload = apply_module1_ablation(
            graph_payload_raw, graph_module1_mode
        )
        graph_ablation_ms = (time.perf_counter() - graph_ablation_t0) * 1000.0

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

        cache_check_ms = 0.0
        cache_store_ms = 0.0
        segment_ms = 0.0
        cache_hit = False
        with self._room_lock:
            cache_check_t0 = time.perf_counter()
            force_stable = bool(request.get("force_stable"))
            cached, cache_key, cache_categories = self._load_room_topology_cache(
                room_occupancy,
                topology_revision=topology_revision,
                epoch=epoch,
                force_stable=force_stable,
            )
            cache_check_ms = (time.perf_counter() - cache_check_t0) * 1000.0
            if cached is not None:
                room_ids = cached["room_ids"]
                room_conf = cached["room_conf"]
                room_merges = {}
                cache_hit = True
            else:
                # A cache entry describes the previous settled state.  Once a
                # full segmentation runs, discard it until the new temporal
                # state has settled and can be cached again.
                self._room_topology_cache = None
                segment_t0 = time.perf_counter()
                room_ids, room_conf = self._segment_rooms_from_occupancy(
                    room_occupancy,
                    force_stable=force_stable,
                )
                room_merges = self.room_segmenter.consume_confirmed_merges()
                segment_ms = (time.perf_counter() - segment_t0) * 1000.0
                cache_store_t0 = time.perf_counter()
                self._store_room_topology_cache(
                    room_occupancy,
                    topology_revision=topology_revision,
                    epoch=epoch,
                    force_stable=force_stable,
                    room_ids=room_ids,
                    room_conf=room_conf,
                    room_merges=room_merges,
                    cache_key=cache_key,
                    cache_categories=cache_categories,
                )
                cache_store_ms = (time.perf_counter() - cache_store_t0) * 1000.0

        crop_t0 = time.perf_counter()
        # A settled topology can be reused for many header-only OCC updates.
        # Keep the already encoded int8 visualization payload on the cache
        # entry so this path does not rebuild a full NumPy raster or allocate a
        # Python list on every update.  Header/info are still rebuilt from the
        # current raw source below, preserving the existing source timestamp,
        # sequence, frame, and geometry semantics.
        cached_room_grid_data = (
            cached.get("encoded_grid_data")
            if isinstance(cached, dict)
            else None
        )
        room_grid = self._build_room_segment_grid(
            room_ids,
            raw=raw,
            encoded_data=cached_room_grid_data,
        )
        if cache_key is not None:
            with self._room_lock:
                cache_entry = getattr(self, "_room_topology_cache", None)
                cached_data_length = -1
                if isinstance(cache_entry, dict):
                    try:
                        cached_data_length = len(
                            cache_entry.get("encoded_grid_data") or []
                        )
                    except TypeError:
                        cached_data_length = -1
                expected_data_length = int(getattr(raw.info, "width", 0)) * int(
                    getattr(raw.info, "height", 0)
                )
                if (
                    isinstance(cache_entry, dict)
                    and cache_entry.get("key") == cache_key
                    and cached_data_length != expected_data_length
                ):
                    # The list is treated as immutable after publication.  A
                    # cache hit can then assign the same payload object to a
                    # fresh ROS message while retiming only its header/info.
                    cache_entry["encoded_grid_data"] = room_grid.data
        crop_ms = (time.perf_counter() - crop_t0) * 1000.0

        # A newer OCC may have arrived while the CPU-heavy room work ran.
        # Never commit that old source when its ternary topology changed.
        commit_raw, commit_room_grid, commit_input_revision, commit_status = (
            self._resolve_room_refresh_commit_source(
                raw,
                room_grid,
                input_revision,
                segmentation_content_key,
                epoch=epoch,
                topology_revision=topology_revision,
            )
        )
        if commit_status in {"content_changed", "raced"}:
            self._record_component_timing("room_worker_stale_content", 1.0)
            # Requeue only the newest source and bypass the normal minimum
            # interval.  This is latest-only: any intermediate requests are
            # coalesced by ``_enqueue_room_refresh``.
            self._enqueue_room_refresh(
                force_stable=bool(request.get("force_stable")),
                reason="room_content_superseded",
                epoch=epoch,
                urgent=True,
            )
            self._record_component_timing("room_worker_stale_discard", 0.0)
            return
        if commit_status == "stale_topology":
            self._record_component_timing("room_worker_stale_discard", 0.0)
            return
        if commit_raw is None or commit_room_grid is None:
            self._record_component_timing("room_worker_stale_discard", 0.0)
            return
        if commit_status == "retimed":
            self._record_component_timing("room_worker_retime_latest", 1.0)
            raw = commit_raw
            room_grid = commit_room_grid
            input_revision = int(commit_input_revision)

        commit_t0 = time.perf_counter()
        commit_lock_t0 = time.perf_counter()
        graph_update_ms = 0.0
        graph_cache_hit = False
        committed = False
        room_commit = None
        with self.lock:
            commit_lock_wait_ms = (time.perf_counter() - commit_lock_t0) * 1000.0
            # Do not publish a room grid against a newer raw OCC source.  The
            # worker remains latest-only, but an older result must be discarded
            # before it can overwrite the graph/readiness watermark and strand
            # a newly opened room behind a stale decomposition.
            if (
                epoch == int(self._room_epoch)
                and topology_revision == int(
                    getattr(self, "_room_topology_revision", 0)
                )
                and getattr(self, "latest_occupancy_grid", None) is raw
                and int(getattr(self, "_room_input_revision", 0))
                == int(input_revision)
            ):
                self.latest_room_segment_grid = room_grid
                # A topology-cache hit means the room IDs/geometry are
                # unchanged. Reapplying the full room graph under the mapper
                # lock on every OCC frame was the main source of avoidable
                # callback stalls. Only commit room topology after a real
                # segmentation or merge.
                if not cache_hit:
                    graph_update_t0 = time.perf_counter()
                    self.graph_store.update_room_grid(
                        raw.info,
                        room_ids,
                        room_conf,
                        room_merges=room_merges,
                        geometry_stability_frames=self.room_geometry_stability_frames,
                    )
                    graph_update_ms = (time.perf_counter() - graph_update_t0) * 1000.0
                graph_cache_hit = bool(
                    getattr(self.graph_store, "last_room_grid_cache_hit", False)
                )
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
        self._record_component_timing("room_worker_graph_ablation", graph_ablation_ms)
        self._record_component_timing("room_worker_overlay", overlay_ms)
        self._record_component_timing("room_worker_topology_cache_check", cache_check_ms)
        self._record_component_timing("room_worker_topology_cache_store", cache_store_ms)
        self._record_component_timing("room_worker_topology_cache_hit", 1.0 if cache_hit else 0.0)
        self._record_component_timing("room_worker_segment", segment_ms)
        self._record_component_timing("room_worker_crop", crop_ms)
        self._record_component_timing("room_worker_commit_lock_wait", commit_lock_wait_ms)
        self._record_component_timing("room_worker_graph_update", graph_update_ms)
        self._record_component_timing(
            "room_worker_graph_cache_hit", 1.0 if graph_cache_hit else 0.0
        )
        self._record_component_timing("room_worker_commit", commit_ms)
        self._record_component_timing("room_worker_total", total_ms)
        if room_commit is not None:
            room_commit["worker_total_ms"] = total_ms
            room_commit["room_segment_ms"] = segment_ms
            room_commit["room_topology_cache_hit"] = cache_hit
            room_commit["graph_update_ms"] = graph_update_ms
            room_commit["room_graph_cache_hit"] = graph_cache_hit
            with self.lock:
                # The worker commit itself is the causal hand-off.  Keep a
                # copy under the mapper lock so the next published bundle can
                # prove that its raw source and graph revision are paired.
                if (
                    int(room_commit["epoch"]) == int(self._room_epoch)
                    and int(room_commit["topology_revision"])
                    == int(getattr(self, "_room_topology_revision", 0))
                    and self._room_snapshot_is_current_locked(
                        raw=raw,
                        input_revision=input_revision,
                    )
                ):
                    self._room_last_commit = dict(room_commit)
            # In strict mode the worker is the event that makes the causal
            # bundle available.  Publish this exact source now instead of
            # waiting for the next 5 Hz timer tick.
            self._publish_causal_room_commit(raw, room_grid, room_commit)
        if not committed:
            # A callback can arrive after the source-resolution check but
            # before this short commit lock.  Drop the result and schedule the
            # latest source immediately rather than exposing an old room map.
            self._enqueue_room_refresh(
                force_stable=bool(request.get("force_stable")),
                reason="room_commit_superseded",
                epoch=epoch,
                urgent=True,
            )
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

    def _room_snapshot_is_current_locked(self, *, raw, input_revision):
        """Return whether a room result still describes the latest OCC source.

        The room worker is deliberately latest-only.  A high-rate mapper can
        publish another callback while segmentation is running, so revision
        equality is the normal fast path.  Some publishers resend the same
        source frame, however; accepting that exact source avoids starving the
        worker without ever committing a result for a newer map.
        """

        current_revision = int(getattr(self, "_room_input_revision", 0))
        if int(input_revision) == current_revision:
            return True
        current_raw = getattr(self, "latest_occupancy_grid", None)
        return self._same_occupancy_source(
            self._occupancy_source_identity(raw),
            self._occupancy_source_identity(current_raw),
        )

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

    @staticmethod
    def _room_overlay_state_key(overlay):
        """Return the small immutable state that changes room overlay cells.

        Detector visibility and graph timestamps do not affect room cuts.  A
        key based on active portal IDs plus their anchored references lets the
        worker reuse an already materialized raster while still invalidating
        it when a portal opens/closes or its geometry is replaced.
        """

        active_ids = {
            str(value)
            for value in (getattr(overlay, "active_portal_ids", None) or [])
        }
        active_ids.difference_update(
            {
                str(value)
                for value in (getattr(overlay, "pending_portal_ids", None) or [])
            }
        )
        if not bool(getattr(overlay, "enabled", False)) or not active_ids:
            return (False, ())
        references = []
        reference_aabbs = {
            str(key): value
            for key, value in (getattr(overlay, "reference_aabbs", None) or {}).items()
        }
        for node_id in sorted(active_ids):
            reference = reference_aabbs.get(node_id)
            if reference is None:
                references.append((node_id, None))
                continue
            try:
                center, size = reference
                references.append(
                    (
                        node_id,
                        tuple(float(value) for value in center[:3]),
                        tuple(float(value) for value in size[:3]),
                    )
                )
            except (TypeError, ValueError, IndexError):
                references.append((node_id, repr(reference)))
        return (
            True,
            tuple(references),
            round(float(getattr(overlay, "clear_padding_m", 0.0)), 9),
            round(float(getattr(overlay, "max_aperture_thickness_m", 0.0)), 9),
        )

    def _room_segmentation_occupancy_from_snapshot(self, raw, graph_payload):
        if raw is None or not self.room_segment_use_semantic_overlay:
            return raw
        overlay = self._room_segmentation_overlay
        overlay.update_graph(graph_payload)
        if not overlay.has_active_portals(include_pending=False):
            # The room worker only reads this message.  Avoid materialising a
            # full data copy plus all-zero clear mask when no *confirmed*
            # portal can affect room topology; pending planning clears remain
            # intentionally excluded from this path.
            self._room_overlay_cache = None
            return raw

        # ``apply`` starts with ``[int(value) for value in raw_data]``.  On a
        # large map that copy dominates room refresh time, even when the raw
        # map and confirmed doorway geometry are unchanged.  Keep an owned
        # source list so in-place bridge buffers are still detected safely;
        # list equality is implemented in C and is substantially cheaper than
        # re-materialising the effective raster.  Header/stamp changes are
        # handled by retiming the cached payload below.
        overlay_key = self._room_overlay_state_key(overlay)
        raw_data = getattr(raw, "data", ())
        cache = getattr(self, "_room_overlay_cache", None)
        if (
            isinstance(cache, dict)
            and cache.get("geometry") == self._occupancy_geometry_key(raw)
            and cache.get("overlay_key") == overlay_key
            and self._same_occupancy_data(cache.get("raw_data"), raw_data)
        ):
            return self._retime_cached_grid(cache.get("effective_data"), raw)

        effective_data, _mask_data, _stats = overlay.apply(
            raw.info,
            raw_data,
            include_pending=False,
        )
        try:
            owned_raw_data = list(raw_data)
        except (TypeError, ValueError):
            owned_raw_data = None
        self._room_overlay_cache = {
            "geometry": self._occupancy_geometry_key(raw),
            "overlay_key": overlay_key,
            "raw_data": owned_raw_data,
            "effective_data": effective_data,
        }
        return self._retime_cached_grid(effective_data, raw)

    def object_callback(self, msg):
        """Queue only the newest detector receipt.

        A few focused tests instantiate the node without ``__init__`` and
        call this callback directly; retain that synchronous compatibility
        path while production nodes use the worker above.
        """

        condition = getattr(self, "_object_work_condition", None)
        worker = getattr(self, "_object_worker_thread", None)
        if condition is None or worker is None:
            return self._process_object_message(msg)
        payload = getattr(msg, "data", msg)
        with condition:
            if self._object_worker_stopping:
                return
            self._object_work_pending = str(payload)
            condition.notify()

    def _process_object_message(self, msg):
        if not self.enable_object_mapping:
            return
        data = getattr(msg, "data", msg)
        parsed = parse_json_object_or_text(data)
        detections = parsed.get("detections")
        if detections is None:
            detections = parse_json_list(data)
        if not isinstance(detections, list):
            return
        # YOLO may keep a 2-D overlay detection when the per-frame 3-D
        # geometry budget is exhausted.  Such records intentionally have no
        # position/size and must never become semantic-map tracks (the
        # legacy store otherwise materializes them at [0, 0, 0]).
        detections = [
            item
            for item in detections
            if isinstance(item, dict)
            and not bool(item.get("geometry_skipped", False))
        ]
        # Keep processing an empty filtered frame.  ObjectMapStore uses the
        # receipt to advance miss/stale bookkeeping, while confirmed graph
        # nodes remain persistent according to their configured lifetime.
        stamp = self._stamp_from_detection_payload(parsed)
        # Keep the relatively expensive detector-to-graph stages visible in
        # the same timing stream as the OCC/room workers.  The values are
        # accumulated while holding ``self.lock`` and emitted after the lock
        # is released, so the diagnostic logger itself cannot extend the
        # critical section or perturb callback ordering.
        object_store_update_ms = 0.0
        object_store_tracking_ms = 0.0
        object_store_total_ms = 0.0
        graph_update_ms = 0.0
        graph_prune_rooms_ms = 0.0
        graph_ensure_rooms_ms = 0.0
        graph_prune_stale_ms = 0.0
        graph_prune_ensure_total_ms = 0.0
        with self.lock:
            # Bind the mirror to the same experiment as its observations;
            # reset may rotate the stream while serialization runs unlocked.
            tracked_stream_epoch = getattr(self, "tracked_stream_epoch", "mapper-legacy")
            room_epoch = int(getattr(self, "_room_epoch", 0))
            object_store_total_t0 = time.perf_counter()
            object_store_t0 = time.perf_counter()
            if self.object_store.update(detections, stamp) is False:
                # Replayed/out-of-order capture cannot increase either the
                # tracker count or the downstream graph's two-frame gate.
                return
            object_store_update_ms = (time.perf_counter() - object_store_t0) * 1000.0
            object_store_tracking_t0 = time.perf_counter()
            tracked_detections = self.object_store.as_tracked_detections(
                min_observations=self.graph_min_observations,
                confirmed_only=False,
                # Keep confirmed semantic objects in the graph/M1 stream when
                # a detector misses a frame.  Visibility is represented by the
                # graph node's is_currently_visible flag; removing the record
                # here made interaction state flicker with YOLO dropouts.
                # Preserve missed objects inside ObjectMapStore/graph state,
                # but only feed detections observed in this receipt into the
                # graph update. Re-emitting every historical track marked all
                # old boxes visible and caused the semantic graph to accumulate
                # a large number of duplicate-looking boxes.
                currently_observed_only=True,
            )
            object_store_tracking_ms += (
                time.perf_counter() - object_store_tracking_t0
            ) * 1000.0
            graph_detections = list(tracked_detections)
            tentative_graph_count = 0
            if (
                getattr(
                    self,
                    "open_refrigerator_content_mapping_enabled",
                    False,
                )
                and callable(
                    getattr(
                        self.graph_store,
                        "has_confirmed_open_refrigerator",
                        None,
                    )
                )
                    and bool(self.graph_store.has_confirmed_open_refrigerator())
            ):
                object_store_tracking_t0 = time.perf_counter()
                tentative_detections = self.object_store.as_tracked_detections(
                    min_observations=getattr(
                        self,
                        "open_refrigerator_content_min_observations",
                        1,
                    ),
                    confirmed_only=False,
                    currently_observed_only=True,
                    ignore_class_confirmations=bool(
                        getattr(
                            self,
                            "open_refrigerator_content_ignore_class_confirmations",
                            True,
                        )
                    ),
                )
                strict_track_ids = {
                    str(item.get("track_id") or item.get("instance_id") or "")
                    for item in tracked_detections
                }
                for detection in tentative_detections:
                    track_id = str(
                        detection.get("track_id")
                        or detection.get("instance_id")
                        or ""
                    )
                    if track_id in strict_track_ids:
                        continue
                    admitted = dict(detection)
                    admitted["tracking_confirmed"] = False
                    admitted[
                        "graph_admission_source"
                    ] = "open_refrigerator_exposure"
                    graph_detections.append(admitted)
                    tentative_graph_count += 1
                object_store_tracking_ms += (
                    time.perf_counter() - object_store_tracking_t0
                ) * 1000.0
                if tentative_graph_count:
                    rospy.loginfo_throttle(
                        2.0,
                        "[semantic_mapping_node] admitting %d tentative "
                        "refrigerator-content track(s) to graph only",
                        tentative_graph_count,
                    )
            object_store_total_ms = (
                time.perf_counter() - object_store_total_t0
            ) * 1000.0
            observations = [
                observation_from_detection(det, observation_id=f"det_{index:04d}")
                for index, det in enumerate(graph_detections, start=1)
            ]
            graph_update_t0 = time.perf_counter()
            self.graph_store.update_observations(
                observations,
                stamp=stamp,
                source_mode="detector_online",
            )
            graph_update_ms = (time.perf_counter() - graph_update_t0) * 1000.0
            # Remove synthetic door-side rooms that were created before the
            # portal passed the two-frame + M1 gate.  Without this cleanup,
            # old provisional rectangles remain in the graph and overlap the
            # real OCC room when the detector changes its door hypothesis.
            prune_rooms = getattr(
                self.graph_store, "prune_unqualified_provisional_rooms", None
            )
            graph_prune_ensure_t0 = time.perf_counter()
            graph_prune_rooms_t0 = time.perf_counter()
            if callable(prune_rooms):
                prune_rooms()
            graph_prune_rooms_ms = (time.perf_counter() - graph_prune_rooms_t0) * 1000.0
            # The room worker can rebuild graph nodes concurrently with a
            # detector callback. Snapshot the values before iterating so
            # a portal merge/prune cannot raise ``dictionary changed size``
            # and drop the current detection update.
            graph_nodes = getattr(self.graph_store, "nodes", {})
            graph_ensure_rooms_t0 = time.perf_counter()
            ensure_room = getattr(
                self.graph_store, "ensure_provisional_room_for_portal", None
            )
            if callable(ensure_room):
                for graph_node in list(graph_nodes.values()):
                    if getattr(graph_node, "type", None) == "portal":
                        ensure_room(graph_node)
            graph_ensure_rooms_ms = (time.perf_counter() - graph_ensure_rooms_t0) * 1000.0
            prune_stale = getattr(self.graph_store, "prune_stale_nodes", None)
            graph_prune_stale_t0 = time.perf_counter()
            if callable(prune_stale):
                prune_stale(
                    getattr(self, "object_stale_after_sec", 0.0), now=stamp
                )
            graph_prune_stale_ms = (time.perf_counter() - graph_prune_stale_t0) * 1000.0
            graph_prune_ensure_total_ms = (
                time.perf_counter() - graph_prune_ensure_t0
            ) * 1000.0
        self._record_component_timing("object_store_update", object_store_update_ms)
        self._record_component_timing(
            "object_store_as_tracked_detections", object_store_tracking_ms
        )
        self._record_component_timing("object_store_total", object_store_total_ms)
        self._record_component_timing("graph_update_observations", graph_update_ms)
        self._record_component_timing(
            "graph_prune_unqualified_rooms", graph_prune_rooms_ms
        )
        self._record_component_timing("graph_ensure_provisional_rooms", graph_ensure_rooms_ms)
        self._record_component_timing("graph_prune_stale_nodes", graph_prune_stale_ms)
        self._record_component_timing(
            "graph_prune_ensure_total", graph_prune_ensure_total_ms
        )
        # JSON compaction and ROS serialization can be sizeable when masks or
        # packed segment evidence are present.  Keep both operations outside
        # the mapper lock so an OCC callback is not blocked by the tracked
        # detection mirror; the object worker is the sole writer of these
        # local records, so the snapshot above remains coherent.
        tracked_payload = {
            "episode_id": tracked_stream_epoch,
            "stamp_sec": stamp,
            "capture_step": parsed.get("capture_step", parsed.get("seq")),
            "detections": tracked_detections,
        }
        # Preserve source-image identity and capture-time pose for M1.
        # ROS Header.seq is rewritten independently on each topic.
        for key in ("stamp_sec", "stamp_nsec", "capture_stamp_sec", "capture_stamp_nsec",
                    "image_sequence", "source_image_sequence", "image_size", "observation_pose_xyyaw"):
            if key in parsed:
                tracked_payload[key] = parsed[key]
        self.tracked_detections_pub.publish(
            String(data=dumps_compact(tracked_payload))
        )
        # Portal-hint state is worker-owned.  Updating it can wait behind an
        # in-flight segmentation, but it must never make every mapper callback
        # wait on that segmentation while holding ``self.lock``.
        portal_structure_changed = self._update_room_portal_hints(
            observations,
            source_mode="detector_online",
            epoch=room_epoch,
        )
        if portal_structure_changed:
            with self.lock:
                self._mark_room_inputs_dirty_locked(structural=True)
            self._enqueue_room_refresh(reason="object_portal_hint")
        # The 10 Hz publish timer observes the new graph revision and emits a
        # coherent bundle. Publishing the complete map synchronously from the
        # detector callback duplicated that work and blocked subsequent YOLO
        # receipts.

    def scene_callback(self, msg):
        if not self.enable_scene_mapping:
            return
        if not getattr(self, "subscribe_pointcloud", True):
            self._warn_scene_callback_without_pointcloud()
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

    @staticmethod
    def _warn_scene_callback_without_pointcloud():
        """Warn once per throttle interval instead of silently losing scene data."""

        message = (
            "[semantic_mapping_node.py] ignoring scene_attribute because "
            "semantic_map/subscribe_pointcloud=false; enable the PointCloud2 "
            "subscription for legacy scene mapping"
        )
        logwarn_throttle = getattr(rospy, "logwarn_throttle", None)
        if callable(logwarn_throttle):
            logwarn_throttle(5.0, message)
            return
        logwarn = getattr(rospy, "logwarn", None)
        if callable(logwarn):
            logwarn(message)

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
        stamp = observation_stamp_seconds(parsed, math.nan)
        if not math.isfinite(stamp):
            stamp = rospy.Time.now().to_sec()
        episode_reset_requested = False
        portal_structure_changed = False
        with self.lock:
            episode_changed = episode_id and episode_id != self.graph_store.episode_id
            episode_reset_requested = bool(parsed.get("episode_reset")) or bool(episode_changed)
            if episode_reset_requested:
                self._save_episode_graph_locked(final=True)
                self.graph_store.reset(episode_id=episode_id, source_mode="realtime_gt_observation")
                self._room_mllm_committed_signatures = {}
                self._room_mllm_committed_at = {}
                self._room_mllm_episode_id = episode_id
                self._room_mllm_selection_cursor = 0
                self._post_open_planning_refresh_after_stamp_sec = None
                self._post_open_room_refresh_result = None
                self.pending_interaction_commands.clear()
                self.object_store.reset()
                self.tracked_stream_epoch = f"mapper_{os.getpid()}_{time.time_ns()}"
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
            room_epoch = int(getattr(self, "_room_epoch", 0))
        # Reset and portal hints are worker-owned state.  Both operations are
        # deliberately outside the mapper lock so an in-flight segmentation
        # cannot stall incoming OCC or graph callbacks.
        if episode_reset_requested:
            self._reset_room_worker_state()
            self._reset_planning_overlay_state()
        portal_structure_changed = self._update_room_portal_hints(
            observations,
            source_mode="realtime_gt_observation",
            epoch=room_epoch,
        )
        if portal_structure_changed:
            with self.lock:
                self._mark_room_inputs_dirty_locked(structural=True)
        if episode_reset_requested or portal_structure_changed:
            self._enqueue_room_refresh(reason="gt_observation", epoch=room_epoch)
        self._collect_and_publish_bundle()

    def interaction_command_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        opening = str(parsed.get("action") or "").casefold() == "open"
        with self.lock:
            if not self._interaction_episode_matches_locked(parsed):
                return
            self._remember_interaction_command_locked(parsed)
            # An optimistic planning clear is reserved for a graph-qualified
            # portal.  A container can be successfully opened without ever
            # altering reachability, so treating every ``open`` command as a
            # pending doorway was able to erase a fridge from OCC.
            pending_portal = opening and self._result_references_topology_portal_locked(
                parsed
            )
        changed = (
            self._set_planning_interaction_pending(
                parsed.get("node_id"), True, node_type="portal"
            )
            if pending_portal
            else False
        )
        if changed:
            self._collect_and_publish_bundle()

    def interaction_result_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        deferred_room_refresh_result = None
        refresh_room_immediately = False
        pending_node_id = ""
        with self.lock:
            if not self._interaction_episode_matches_locked(parsed):
                return
            if getattr(self, "require_interaction_command_id", False) and not parsed.get("command_id"):
                return
            try:
                command = self._take_interaction_command_locked(parsed)
            except ValueError as exc:
                rospy.logwarn_throttle(5.0, "Rejected interaction result: %s", exc)
                return
            # If a producer supplied an explicit command id, an unknown id is
            # a stale/replayed event, not permission to infer the target from
            # its object name.  Opaque legacy results without command_id keep
            # the existing identity fallback in the graph store.
            if str(parsed.get("command_id") or "") and command is None:
                return
            parsed = merge_interaction_result_with_command(parsed, command)
            if not self._interaction_episode_matches_locked(parsed):
                return
            stamp_value = parsed.get("stamp_sec")
            stamp = (
                float(stamp_value)
                if stamp_value is not None
                else rospy.Time.now().to_sec()
            )
            pending_node_id = str(parsed.get("node_id") or "")
            changed = self.graph_store.update_interaction_result(parsed, stamp=stamp)
            successful_open = self._is_successful_open_result(parsed)
            portal_reference = (
                self._confirmed_open_portal_reference_locked(parsed)
                if successful_open
                and self._result_references_topology_portal_locked(parsed)
                else None
            )
            # The stable portal reference is the topology authority.  Do not
            # arm raw-OCC/room work solely because a result carried
            # ``node_type=portal``: a stale M1 type patch can otherwise turn a
            # container result into a doorway clear.
            is_portal_open = successful_open and portal_reference is not None
            if is_portal_open:
                # This must not depend on ``changed``.  A result can arrive
                # after the graph has already been updated, while the next
                # raw OCC is still the first map that can prove post-open
                # traversability to the planner.
                self._post_open_planning_refresh_after_stamp_sec = stamp
                self._post_open_room_refresh_result = dict(parsed)
                # A room split is a graph/topology product, not a costmap
                # readiness gate. Refresh it immediately from the latest raw
                # grid as soon as a confirmed portal result arrives; the next
                # OCC frame still drives the causal planning-map bridge.
                refresh_room_immediately = (
                    getattr(self, "latest_occupancy_grid", None) is not None
                )
                if refresh_room_immediately:
                    deferred_room_refresh_result = dict(parsed)
                    self._post_open_room_refresh_result = None
                rospy.loginfo(
                    "[semantic_mapping_node.py] armed post-open raw OCC bridge: "
                    "result_stamp=%.6f node_id=%s node_type=%s "
                    "graph_changed=%s portal_reference=%s",
                    stamp,
                    str(parsed.get("node_id") or ""),
                    str(parsed.get("node_type") or ""),
                    changed,
                    True,
                )
        pending_changed = self._set_planning_interaction_pending(
            pending_node_id, False
        )
        if refresh_room_immediately:
            self._enqueue_room_refresh(
                force_stable=bool(getattr(self, "room_post_open_force_refresh", True)),
                post_open_result=deferred_room_refresh_result,
                reason="post_open_result",
            )
        if (changed or pending_changed) and not is_portal_open:
            self._collect_and_publish_bundle()

    def _interaction_episode_matches_locked(self, payload):
        episode = str(payload.get("episode_id") or "")
        current = str(getattr(self.graph_store, "episode_id", "") or "")
        return not (episode and current and episode != current)

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
        stamp_value = parsed.get("stamp_sec")
        stamp = (
            float(stamp_value)
            if stamp_value is not None
            else rospy.Time.now().to_sec()
        )
        changed = False
        with self.lock:
            accepted_epochs = {
                str(self.graph_store.episode_id or ""),
                str(getattr(self, "tracked_stream_epoch", "") or ""),
            } - {""}
            if episode_id and accepted_epochs and episode_id not in accepted_epochs:
                return
            for patch in parsed.get("updates") or []:
                if isinstance(patch, dict):
                    accepted = self.graph_store.apply_attribute_patch(patch, stamp=stamp)
                    if accepted:
                        self._sync_confirmed_m1_track_label_locked(patch)
                    changed = accepted or changed
            prune_rooms = getattr(
                self.graph_store, "prune_unqualified_provisional_rooms", None
            )
            if callable(prune_rooms):
                prune_rooms()
        if changed:
            self._collect_and_publish_bundle()

    def _sync_confirmed_m1_track_label_locked(self, patch):
        """Sync the graph's accepted name, never an unvalidated model proposal."""
        object_id = str(patch.get("object_id") or "")
        nodes = getattr(self.graph_store, "nodes", {})
        node = nodes.get(object_id)
        if node is None:
            node = next((candidate for candidate in nodes.values()
                         if self.graph_store._node_matches_identity(candidate, object_id)), None)
        if node is None:
            return
        attributes = node.attributes
        if (not attributes.get("m1_name_override")
            or str(attributes.get("m1_observed_object_name") or "") != node.label):
            return
        track_id = str(attributes.get("instance_id") or "")
        if track_id:
            self.object_store.set_m1_canonical_label(track_id, node.label)

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
                    room_key = str(patch.get("room_node_id") or "")
                    if not room_key and patch.get("room_id") is not None:
                        try:
                            room_key = f"room_{int(patch.get('room_id'))}"
                        except (TypeError, ValueError):
                            room_key = ""
                    status = str(
                        patch.get("room_attribute_status")
                        or patch.get("attribute_status")
                        or ""
                    ).casefold()
                    # Rule fallback is useful immediately, but it must not
                    # clear the dirty bit: the same room will be retried after
                    # the failure refresh interval.  Only an actual MLLM
                    # result commits the evidence signature.
                    applied = self.graph_store.apply_room_attribute_patch(
                        patch, stamp=stamp
                    )
                    if (
                        status == "ready"
                        and not bool(patch.get("fallback"))
                        and room_key
                        and applied
                    ):
                        signature = str(patch.get("observation_signature") or "")
                        if signature:
                            committed = getattr(
                                self, "_room_mllm_committed_signatures", None
                            )
                            if committed is None:
                                committed = {}
                                self._room_mllm_committed_signatures = committed
                            committed[room_key] = signature
                            self._room_mllm_committed_at[room_key] = time.monotonic()
                    changed = applied or changed
        if changed:
            self._collect_and_publish_bundle()

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
        received_mono_s = time.monotonic()
        try:
            source_age_ms = self._occupancy_source_age_ms(msg, rospy.get_time())
        except rospy.exceptions.ROSInitException:
            source_age_ms = None  # Offline harnesses need no ROS clock.
        lock_t0 = time.perf_counter()
        deferred_room_refresh_result = None
        force_stable = False
        post_open_graph_payload_raw = None
        post_open_graph_module1_mode = None
        post_open_result_stamp = None
        source_identity = self._occupancy_source_identity(msg)
        with self.lock:
            room_epoch = int(getattr(self, "_room_epoch", 0))
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
            self.latest_occupancy_grid = msg
            # Capture receipt time next to the raw source pointer.  The room
            # worker may process a coalesced request later, so a global timing
            # counter would otherwise report the newer frame's latency.
            self._latest_occupancy_received_mono_s = received_mono_s
            self._mark_room_inputs_dirty_locked()
            post_open_t0 = time.perf_counter()
            if self._raw_occupancy_is_after_post_open_refresh_locked(msg):
                result_stamp = self._post_open_planning_refresh_after_stamp_sec
                post_open_result_stamp = result_stamp
                # Publish the first source-new map before room segmentation,
                # but apply the already-confirmed portal overlay first.  A raw
                # sensor frame may still contain the old door leaf immediately
                # after force success; publishing it verbatim briefly recloses
                # the planning map and can strand the post-open traversal.
                # Snapshot the graph while locked; Module-1 ablation is
                # performed below after releasing the mapper lock.
                graph_store = getattr(self, "graph_store", None)
                overlay = getattr(self, "semantic_occ_overlay", None)
                if graph_store is not None and overlay is not None:
                    post_open_graph_payload_raw = graph_store.as_graph_dict()
                    post_open_graph_module1_mode = getattr(
                        getattr(self, "ablation", None), "module1", "full"
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
        # Keep the causal post-open publication ahead of room work, but do the
        # potentially large Module-1 deepcopy and planning materialization
        # outside ``self.lock`` so an OCC callback never blocks graph updates.
        if post_open_graph_payload_raw is not None:
            graph_ablation_t0 = time.perf_counter()
            post_open_graph_payload = apply_module1_ablation(
                post_open_graph_payload_raw,
                post_open_graph_module1_mode,
            )
            self._publish_post_open_raw_planning_grid_locked(
                msg,
                result_stamp=post_open_result_stamp,
                graph_payload=post_open_graph_payload,
            )
            post_open_ms += (time.perf_counter() - graph_ablation_t0) * 1000.0
        enqueue_t0 = time.perf_counter()
        self._enqueue_room_refresh(
            force_stable=force_stable,
            post_open_result=(
                deferred_room_refresh_result if force_stable else None
            ),
            reason="post_open" if deferred_room_refresh_result is not None else "occupancy",
            source=source_identity,
            epoch=room_epoch,
        )
        room_enqueue_ms = (time.perf_counter() - enqueue_t0) * 1000.0
        total_ms = (time.perf_counter() - callback_t0) * 1000.0
        self._record_component_timing("occupancy_lock_wait", lock_wait_ms)
        self._record_component_timing("occupancy_post_open_bridge", post_open_ms)
        self._record_component_timing("occupancy_scene_init", scene_init_ms)
        self._record_component_timing("occupancy_room_enqueue", room_enqueue_ms)
        self._record_component_timing("occupancy_callback_total", total_ms)
        if source_age_ms is not None:
            self._record_component_timing("occupancy_source_age_at_callback", source_age_ms)

    @staticmethod
    def _is_successful_open_result(result):
        action = str(result.get("action") or "").strip().casefold()
        observation_only = (
            str(result.get("observation_outcome") or "").strip().casefold()
            == "finish_without_action"
        )
        if result.get("success") is not True or (
            action not in {"", "open"} or (not action and not observation_only)
        ):
            return False
        state = str(
            result.get("state") or result.get("post_state") or ""
        ).strip().casefold()
        capability = str(
            result.get("interaction_capability")
            or result.get("capability")
            or ""
        ).strip().casefold()
        source = str(result.get("source") or "").strip().casefold()
        if (
            observation_only
            and state in {"open", "opened", "ajar", "static_open"}
            and isinstance(result.get("portal_aperture_evidence"), dict)
        ):
            # An already-open doorway still needs the same post-observation
            # OCC/room refresh as a force-open result.  Without this bridge the
            # graph says open while planning OCC waits forever and the candidate
            # is recreated on the next graph revision.
            return True
        # ``static_open`` is an immediate capability result, not a physical
        # transition.  Do not arm the post-open raw-OCC bridge or force a room
        # refresh for it; ordinary occupancy updates remain responsible for
        # discovering any traversable space beyond the opening.
        return not (
            state in {"static", "static_open", "static_closed"}
            or capability in {
                "static",
                "blocked",
                "unsupported",
                "unavailable",
                "locked",
            }
            or source == "executor_static_portal"
        )

    def _publish_post_open_raw_planning_grid_locked(
        self, planning_grid, *, result_stamp, graph_payload=None
    ):
        """Publish the first post-open OCC with the confirmed portal cleared."""

        effective_grid = planning_grid
        overlay_stats = {"active_portal_ids": [], "cleared_cells": 0}
        overlay = getattr(self, "semantic_occ_overlay", None)
        if overlay is not None:
            # ``graph_payload`` is snapshotted and ablated by the caller
            # outside ``self.lock``.  Minimal/replay callers may omit it; an
            # existing cache is still preferable to stalling the first fresh
            # OCC frame on a synchronous graph deepcopy.
            if graph_payload is None:
                graph_payload = getattr(self, "_graph_payload_cache", None) or {}
            (
                effective_grid,
                planning_update,
                door_clear_mask,
                overlay_stats,
            ) = self._build_planning_products_from_snapshot(
                planning_grid,
                graph_payload,
            )
        self.planning_occupancy_grid_pub.publish(effective_grid)
        if overlay is not None:
            if door_clear_mask is not None:
                self.door_clear_mask_pub.publish(door_clear_mask)
            if planning_update is not None:
                self.planning_occupancy_grid_updates_pub.publish(planning_update)
        raw_stamp = self._occupancy_header_stamp_sec(planning_grid)
        rospy.loginfo(
            "[semantic_mapping_node.py] published post-open effective planning OCC: "
            "raw_stamp=%s result_stamp=%s raw_seq=%s active_portals=%s cleared=%s",
            "%.6f" % raw_stamp if raw_stamp is not None else "missing",
            "%.6f" % result_stamp if result_stamp is not None else "missing",
            int(getattr(planning_grid.header, "seq", 0) or 0),
            list(overlay_stats.get("active_portal_ids") or []),
            int(overlay_stats.get("cleared_cells", 0) or 0),
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

    @staticmethod
    def _occupancy_source_age_ms(msg, now_sec):
        stamp = SemanticMappingNode._occupancy_header_stamp_sec(msg)
        if stamp is None:
            return None
        try:
            elapsed = (float(now_sec) - stamp) * 1000.0
        except (TypeError, ValueError, OverflowError):
            return None
        return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else None

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

    @staticmethod
    def _node_is_topology_portal(node):
        """Return whether a graph node may affect occupancy topology.

        ``node.type`` can be an MLLM-facing semantic label.  A source-observed
        container is never allowed to gain portal topology from one delayed
        attribute patch.  Legacy graph payloads without provenance retain their
        original portal behavior for replay compatibility.
        """

        attributes = (node or {}).get("attributes") or {}
        source_type = str(
            attributes.get("topology_type")
            or attributes.get("observation_node_type")
            or ""
        ).strip().casefold()
        # The source topology is immutable evidence.  A delayed M1 patch may
        # temporarily change the public semantic type (for example a door
        # crop answered as ``object``) before the sealed interaction result
        # arrives; that must not suppress the post-open raw-OCC bridge.  The
        # converse is equally important: a source bottle/container answered
        # as ``portal`` must never be allowed to alter occupancy topology.
        if source_type:
            return source_type == "portal"
        return str((node or {}).get("type") or "").casefold() == "portal"

    def _result_references_topology_portal_locked(self, result):
        """Match a command/result only to a source-qualified portal node."""

        requested_node_id = str((result or {}).get("node_id") or "")
        requested_object_id = str(
            (result or {}).get("object_id")
            or (result or {}).get("instance_id")
            or ""
        )
        for node in self.graph_store.as_graph_dict().get("nodes") or []:
            if not self._node_is_topology_portal(node):
                continue
            attributes = node.get("attributes") or {}
            instance_id = str(attributes.get("instance_id") or "")
            source_object_name = str(attributes.get("source_object_name") or "")
            if requested_node_id and str(node.get("id") or "") == requested_node_id:
                return True
            if requested_object_id and requested_object_id in {
                instance_id,
                source_object_name,
            }:
                return True
        return False

    def _confirmed_open_portal_reference_locked(self, result):
        """Build a stable, closed-door geometry observation for a result."""

        requested_node_id = str(result.get("node_id") or "")
        requested_object_id = str(
            result.get("object_id") or result.get("instance_id") or ""
        )
        for node in self.graph_store.as_graph_dict().get("nodes") or []:
            if not self._node_is_topology_portal(node):
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
            if state not in {"open", "opened", "ajar", "static_open"}:
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
        if not self.semantic_occ_overlay.has_active_portals(include_pending=False):
            return raw
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
        self.latest_room_segment_grid = self._build_room_segment_grid(room_ids)
        self.graph_store.update_room_grid(
            self.latest_occupancy_grid.info,
            room_ids,
            room_conf,
            room_merges=room_merges,
            geometry_stability_frames=self.room_geometry_stability_frames,
        )

    def _publish_causal_room_commit(self, raw, room_grid, room_commit):
        """Publish one exact room commit immediately in strict mode.

        The normal timer remains responsible for low-priority visual products.
        This path builds a source-pinned core bundle from the worker's ``raw``
        and ``room_grid`` references, so a newer OCC callback cannot silently
        replace the source between room commit and readiness.
        """

        if not getattr(self, "step_ready_require_room_graph", False):
            return
        if raw is None or room_grid is None or not room_commit:
            return
        publish_lock = getattr(self, "_publish_lock", None)
        if publish_lock is None:
            return self._publish_causal_room_commit_locked(raw, room_grid, room_commit)
        with publish_lock:
            return self._publish_causal_room_commit_locked(raw, room_grid, room_commit)

    def _publish_causal_room_commit_locked(self, raw, room_grid, room_commit):
        with self.lock:
            current = dict(getattr(self, "_room_last_commit", None) or {})
            if not self._same_occupancy_source(
                current.get("source"), room_commit.get("source")
            ):
                return False
            # Snapshot mutable graph state while locked; defer the potentially
            # large Module-1 deepcopy until after releasing the mapper lock.
            graph_payload_raw = self.graph_store.as_graph_dict()
            graph_module1_mode = self.ablation.module1
            snapshot = {
                "obj_map": None,
                "scene_info": None,
                "scene_data": None,
                "scene_confidence_data": None,
                "scene_revision": 0,
                "publish_scene_grids": False,
                "room_segment_grid": room_grid,
                "room_commit": dict(room_commit),
                "graph_payload_raw": graph_payload_raw,
                "graph_module1_mode": graph_module1_mode,
                "graph_cache_token": (
                    int(getattr(self.graph_store, "graph_revision", -1)),
                    graph_module1_mode,
                ),
                "raw_occupancy_grid": raw,
                "causal_core_only": True,
            }
        build_t0 = time.perf_counter()
        bundle = self._build_publish_bundle_from_snapshot(snapshot)
        build_ms = (time.perf_counter() - build_t0) * 1000.0
        publish_t0 = time.perf_counter()
        try:
            self._publish_bundle(bundle)
        except rospy.ROSException as exc:
            if "closed topic" not in str(exc).lower() and not rospy.is_shutdown():
                raise
        publish_ms = (time.perf_counter() - publish_t0) * 1000.0
        ready_payload = self._semantic_mapping_ready_payload_for_bundle(bundle)
        self.step_ready_pub.publish(
            String(data=json.dumps(ready_payload, separators=(",", ":")))
        )
        self._record_causal_ready_once(ready_payload)
        self._record_component_timing("room_commit_publish_build", build_ms)
        self._record_component_timing("room_commit_publish_ros", publish_ms)
        self._record_component_timing(
            "room_commit_publish_total", build_ms + publish_ms
        )
        return bool(ready_payload.get("ready"))

    def publish_callback(self, _event):
        if rospy.is_shutdown():
            return
        publish_lock = getattr(self, "_publish_lock", None)
        if publish_lock is not None:
            with publish_lock:
                return self._publish_callback_locked()
        return self._publish_callback_locked()

    def _publish_input_signature_locked(self):
        """Return a cheap token for products visible to subscribers.

        Avoid ``as_graph_dict`` here: it materialises the complete graph and
        would defeat the gate. Store revisions and source identities cover
        the producers that can change the emitted bundle.
        """
        raw = getattr(self, "latest_occupancy_grid", None)
        source = self._occupancy_source_identity(raw)
        if isinstance(source, dict):
            raw_source = (
                int(source.get("step_index", -1) or -1),
                float(source.get("stamp_sec", 0.0) or 0.0),
            )
        else:
            raw_source = None
        info = getattr(raw, "info", None)
        raw_geometry = (
            int(getattr(info, "width", 0) or 0),
            int(getattr(info, "height", 0) or 0),
            float(getattr(info, "resolution", 0.0) or 0.0),
        ) if info is not None else None
        room_commit = getattr(self, "_room_last_commit", None) or {}
        try:
            room_graph_revision = int(room_commit.get("graph_revision", -1))
        except (TypeError, ValueError):
            room_graph_revision = -1
        overlay = getattr(self, "semantic_occ_overlay", None)
        active_portals = tuple(sorted(str(value) for value in (
            getattr(overlay, "active_portal_ids", None) or []
        )))
        return (
            raw_source,
            id(raw),
            raw_geometry,
            # ``graph_store.graph_revision`` also advances for detector-only
            # visibility/box refreshes.  Including it here made the 10-Hz
            # mapper timer rebuild and republish the full planning OCC for
            # every YOLO receipt, even when the raw map and interaction
            # topology were unchanged.  Graph JSON/markers are independently
            # bounded by ``graph_publish_period_sec`` and the heartbeat below;
            # retain the room commit and active portal set as the immediate
            # topology signals needed for causal door overlays.
            room_graph_revision,
            int(getattr(self, "_scene_grid_revision", 0)),
            id(getattr(self, "latest_room_segment_grid", None)),
            active_portals,
            bool(getattr(self, "_planning_overlay_was_active", False)),
        )

    def _publish_callback_locked(self):
        callback_t0 = time.perf_counter()
        lock_t0 = time.perf_counter()
        with self.lock:
            lock_wait_ms = (time.perf_counter() - lock_t0) * 1000.0
            signature = self._publish_input_signature_locked()
            now_mono = time.monotonic()
            heartbeat = float(getattr(self, "publish_heartbeat_period_sec", 0.0))
            previous_signature = getattr(self, "_last_publish_signature", None)
            previous_mono = float(getattr(self, "_last_publish_mono", 0.0))
            if (
                heartbeat > 0.0
                and signature == previous_signature
                and previous_mono > 0.0
                and now_mono - previous_mono < heartbeat
            ):
                self._record_component_timing("publish_callback_skipped", 0.0)
                return False
        collect_t0 = time.perf_counter()
        publish_bundle = self._collect_publish_bundle()
        collect_ms = (time.perf_counter() - collect_t0) * 1000.0
        publish_t0 = time.perf_counter()
        self._safe_publish_bundle(publish_bundle)
        with self.lock:
            self._last_publish_signature = signature
            self._last_publish_mono = time.monotonic()
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
        return True

    def _safe_publish_bundle(self, bundle):
        publish_lock = getattr(self, "_publish_lock", None)
        if publish_lock is None:
            try:
                self._publish_bundle(bundle)
            except rospy.ROSException as exc:
                if "closed topic" not in str(exc).lower() and not rospy.is_shutdown():
                    raise
            return
        with publish_lock:
            try:
                self._publish_bundle(bundle)
            except rospy.ROSException as exc:
                if "closed topic" not in str(exc).lower() and not rospy.is_shutdown():
                    raise

    def _collect_and_publish_bundle(self):
        """Keep snapshot construction and publication in causal order.

        Acquiring ``_publish_lock`` only inside ``_safe_publish_bundle`` is
        too late: two callbacks can build old/new snapshots concurrently and
        then publish them in the reverse order.  Hold the same re-entrant lock
        while collecting the snapshot so an older callback always finishes
        before a newer callback is allowed to collect and publish its state.
        """

        publish_lock = getattr(self, "_publish_lock", None)
        if publish_lock is None:
            bundle = self._collect_publish_bundle()
            self._safe_publish_bundle(bundle)
            return bundle
        with publish_lock:
            bundle = self._collect_publish_bundle()
            self._safe_publish_bundle(bundle)
            return bundle

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

    def _room_robot_xy(self):
        listener = getattr(self, "tf_listener", None)
        if listener is None:
            return None
        try:
            position, _rotation = listener.lookupTransform(
                self.world_frame, self.base_frame, rospy.Time(0)
            )
            return float(position[0]), float(position[1])
        except (IndexError, TypeError, ValueError, tf.Exception):
            return None

    def _build_room_attribute_request_locked(self, graph_payload):
        """Build one changed, nearby room request for Module-1.

        The mapper publishes at most one room per message.  A room is nearby
        when it contains the robot, is currently visible, or is connected to
        an anchor room through a portal.  Only a changed box/member set or an
        expired success refresh is requested.  Failures remain eligible for a
        retry subject to the inference lane's per-room cooldown.
        """

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

        active_rooms = []
        room_records = []
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
            room_box = normalized_room_box(
                node.get("aabb_center"), node.get("aabb_size")
            )
            active_rooms.append({"room_id": room_id, "room_box": room_box})
            evidence = sorted(
                evidence_by_room.get(room_id, []),
                key=lambda item: (item["object_id"], item["node_id"]),
            )
            if len(evidence) < self.room_mllm_min_evidence_objects:
                continue
            room_records.append(
                {
                    "room_id": room_id,
                    "room_node_id": str(node.get("id") or f"room_{room_id}"),
                    "room_box": room_box,
                    "objects": evidence,
                }
            )
        if not room_records:
            return None

        # Derive a small topological neighborhood from portal ``connects`` edges.
        node_by_id = {str(node.get("id") or ""): node for node in nodes}
        adjacency = {int(room["room_id"]): set() for room in active_rooms}
        for edge in graph_payload.get("edges") or []:
            if str(edge.get("relation") or "") != "connects":
                continue
            src = node_by_id.get(str(edge.get("src_id") or ""), {})
            dst = node_by_id.get(str(edge.get("dst_id") or ""), {})
            portal = src if str(src.get("type") or "") == "portal" else dst
            other = dst if portal is src else src
            if str(portal.get("type") or "") != "portal":
                continue
            try:
                connected = [
                    int(value)
                    for value in (
                        (portal.get("attributes") or {}).get("connected_room_ids")
                        or []
                    )
                ]
            except (TypeError, ValueError):
                connected = []
            if not connected:
                try:
                    value = other.get("room_id")
                    connected = [int(value)] if value is not None else []
                except (TypeError, ValueError):
                    connected = []
            for left in connected:
                adjacency.setdefault(left, set()).update(
                    right for right in connected if right != left
                )

        robot_xy = self._room_robot_xy()
        containing_rooms = []
        if robot_xy is not None:
            for room in active_rooms:
                box = room["room_box"]
                center = box["center_xy"]
                size = box["size_xy"]
                if len(center) != 2 or len(size) != 2:
                    continue
                if all(
                    abs(robot_xy[axis] - center[axis]) <= size[axis] / 2.0
                    for axis in (0, 1)
                ):
                    containing_rooms.append((size[0] * size[1], room["room_id"]))
        visible_rooms = {
            int(room["room_id"])
            for room in room_records
            if any(bool(item.get("currently_visible")) for item in room["objects"])
        }
        anchor_rooms = (
            {min(containing_rooms)[1]} if containing_rooms else visible_rooms
        )
        if not anchor_rooms:
            return None
        nearby_rooms = set(anchor_rooms)
        for room_id in anchor_rooms:
            nearby_rooms.update(adjacency.get(room_id, set()))

        changed_rooms = []
        episode_id = str(graph_payload.get("episode_id") or "")
        now = time.monotonic()
        state_lock = getattr(self, "lock", None)
        with state_lock if state_lock is not None else nullcontext():
            if episode_id and episode_id != getattr(self, "_room_mllm_episode_id", ""):
                self._room_mllm_committed_signatures = {}
                self._room_mllm_committed_at = {}
                self._room_mllm_episode_id = episode_id
                self._room_mllm_selection_cursor = 0
            committed_signatures = getattr(self, "_room_mllm_committed_signatures", {})
            completed_at = getattr(self, "_room_mllm_committed_at", {})
            success_refresh_s = getattr(
                self, "room_mllm_success_refresh_interval_s", 120.0
            )
            for room in room_records:
                if int(room["room_id"]) not in nearby_rooms:
                    continue
                key = str(room["room_node_id"])
                signature = room_evidence_signature(
                    room["room_id"], room["room_box"], room["objects"]
                )
                if committed_signatures.get(key) == signature and (
                    success_refresh_s <= 0.0
                    or now - completed_at.get(key, 0.0) < success_refresh_s
                ):
                    continue
                room["observation_signature"] = signature
                changed_rooms.append(room)
            if not changed_rooms:
                return None
            candidates = sorted(
                changed_rooms,
                key=lambda room: (
                    0 if int(room["room_id"]) in anchor_rooms else 1,
                    int(room["room_id"]),
                    str(room["room_node_id"]),
                ),
            )
            selection_cursor = int(getattr(self, "_room_mllm_selection_cursor", 0) or 0)
            selected = candidates[selection_cursor % len(candidates)]
            self._room_mllm_selection_cursor = selection_cursor + 1
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
            # Deliberately keep a single room in every request.  The consumer
            # remains list-compatible for old publishers, but new traffic is
            # isolated per room and cannot make one large request head-of-line
            # block all neighboring rooms.
            "rooms": [selected],
        }

    def _store_graph_payload_cache_locked(self, token, payload):
        """Install a fully ablated graph payload without regressing a newer one.

        The payload is built outside ``self.lock``.  Multiple ROS callbacks can
        therefore finish in either order; a late build must not overwrite a
        cache produced from a newer graph revision.
        """

        if token is None or payload is None:
            return
        existing = getattr(self, "_graph_payload_cache_token", None)
        if (
            isinstance(existing, (tuple, list))
            and len(existing) >= 2
            and isinstance(token, (tuple, list))
            and len(token) >= 2
            and str(existing[1]) == str(token[1])
        ):
            try:
                if int(existing[0]) > int(token[0]):
                    return
            except (TypeError, ValueError):
                pass
        self._graph_payload_cache_token = token
        self._graph_payload_cache = payload

    def _materialize_graph_payload_from_snapshot(self, snapshot):
        """Apply Module-1 policy after the mapper lock has been released."""

        raw_payload = snapshot.get("graph_payload_raw")
        if raw_payload is None:
            return snapshot.get("graph_payload") or {}
        module1_mode = snapshot.get(
            "graph_module1_mode",
            str(getattr(getattr(self, "ablation", None), "module1", "full")),
        )
        graph_payload = apply_module1_ablation(raw_payload, module1_mode)
        token = snapshot.get("graph_cache_token")
        lock = getattr(self, "lock", None)
        if lock is None:
            self._store_graph_payload_cache_locked(token, graph_payload)
        else:
            with lock:
                self._store_graph_payload_cache_locked(token, graph_payload)
        return graph_payload

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
        module1_mode = str(getattr(self.ablation, "module1", "full"))
        try:
            graph_revision = int(getattr(self.graph_store, "graph_revision", -1))
        except (TypeError, ValueError, AttributeError):
            graph_revision = -1
        now_mono = time.monotonic()
        last_graph_mono = float(getattr(self, "_last_graph_publish_mono", 0.0))
        graph_publish_due = bool(
            getattr(self, "_graph_payload_cache", None) is None
            or last_graph_mono <= 0.0
            or now_mono - last_graph_mono >= float(
                getattr(self, "graph_publish_period_sec", 0.2)
            )
        )
        # ``graph_revision`` is intentionally not itself a force-publish
        # signal: detector visibility updates bump it at the camera rate even
        # when the semantic topology is unchanged.  Rebuild the expensive
        # graph JSON/markers at the bounded heartbeat; new objects/state then
        # become visible within at most one graph period.
        cached_token = getattr(self, "_graph_payload_cache_token", None)
        graph_payload_raw = None
        if graph_publish_due or getattr(self, "_graph_payload_cache", None) is None:
            graph_cache_token = (graph_revision, module1_mode)
            # ``as_graph_dict`` must remain under ``self.lock`` because it
            # snapshots mutable graph-store state.  The expensive Module-1
            # ablation/deepcopy is deliberately deferred to the lock-free
            # bundle build below; otherwise a large persistent graph stalls
            # OCC and detector callbacks for the full deepcopy duration.
            graph_payload_raw = self.graph_store.as_graph_dict()
            graph_payload = None
        else:
            graph_cache_token = cached_token
            graph_payload = self._graph_payload_cache
        return {
            "obj_map": (
                object_store.as_obj_map()
                if enable_object_mapping
                and object_store is not None
                and graph_publish_due
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
            "graph_payload_raw": graph_payload_raw,
            "graph_module1_mode": module1_mode,
            "graph_cache_token": graph_cache_token,
            "publish_graph_products": graph_publish_due,
            "raw_occupancy_grid": self.latest_occupancy_grid,
        }

    def _set_planning_interaction_pending(self, node_id, pending, *, node_type=None):
        overlay_lock = getattr(self, "_planning_overlay_lock", None)
        def set_pending():
            try:
                return self.semantic_occ_overlay.set_interaction_pending(
                    node_id, pending, node_type=node_type
                )
            except TypeError:
                # Minimal test doubles predate the optional topology hint.
                return self.semantic_occ_overlay.set_interaction_pending(node_id, pending)
        if overlay_lock is None:
            return set_pending()
        with overlay_lock:
            return set_pending()

    def _reset_planning_overlay_state(self):
        overlay_lock = getattr(self, "_planning_overlay_lock", None)
        if overlay_lock is None:
            self.semantic_occ_overlay.reset()
            self.semantic_occ_update_tracker.reset()
            self._planning_products_cache = None
            return
        with overlay_lock:
            self.semantic_occ_overlay.reset()
            self.semantic_occ_update_tracker.reset()
            self._planning_clear_mask_initialized = False
            self._planning_clear_mask_geometry_key = None
            self._planning_overlay_was_active = False
            self._planning_products_cache = None

    def _build_grid_from_snapshot(self, data, info):
        grid = NumpyOccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.world_frame
        grid.info = info
        grid.data = occupancy_data_snapshot(data)
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
            self.semantic_occ_overlay.update_graph(graph_payload or {})
            geometry_key = self._occupancy_geometry_key(raw)
            overlay_key = self._planning_overlay_key()
            overlay_active = bool(
                self.semantic_occ_overlay.enabled
                and self.semantic_occ_overlay.active_portal_ids
            )
            if not overlay_active:
                previous_active = bool(
                    getattr(self, "_planning_overlay_was_active", False)
                )
                must_publish_zero_mask = (
                    not bool(getattr(self, "_planning_clear_mask_initialized", False))
                    or previous_active
                    or geometry_key
                    != getattr(self, "_planning_clear_mask_geometry_key", None)
                )
                door_clear_mask = None
                zero_mask = None
                if must_publish_zero_mask:
                    cache = getattr(self, "_planning_products_cache", None) or {}
                    zero_mask = cache.get("zero_mask_data")
                    if (
                        cache.get("mode") != "inactive"
                        or cache.get("geometry_key") != geometry_key
                        or zero_mask is None
                        or len(zero_mask) != int(raw.info.width) * int(raw.info.height)
                    ):
                        zero_mask = [0] * (
                            int(raw.info.width) * int(raw.info.height)
                        )
                    door_clear_mask = self._retime_cached_grid(zero_mask, raw)
                    self._planning_clear_mask_initialized = True
                    self._planning_clear_mask_geometry_key = geometry_key
                if previous_active:
                    # A full raw planning map is sent below, so incremental
                    # overlay state is no longer useful after the clear.
                    self.semantic_occ_update_tracker.reset()
                self._planning_overlay_was_active = False
                self._planning_products_cache = {
                    "mode": "inactive",
                    "geometry_key": geometry_key,
                    "overlay_key": overlay_key,
                    "zero_mask_data": zero_mask,
                    "overlay_stats": {
                        "active_portal_ids": [],
                        "cleared_cells": 0,
                        "update_bounds": None,
                        "valid": True,
                    },
                }
                return raw, None, door_clear_mask, {
                    "active_portal_ids": [],
                    "cleared_cells": 0,
                    "update_bounds": None,
                    "valid": True,
                }

            cache = getattr(self, "_planning_products_cache", None) or {}
            raw_data = getattr(raw, "data", None)
            if cache.get("raw_data_ref") is raw_data:
                # A few in-process map producers reuse an immutable data
                # buffer while only advancing the ROS header.  Keep that
                # buffer strongly referenced in the cache so Python cannot
                # recycle its identity between frames.
                raw_data_key = cache.get("raw_data_key")
            elif self._same_occupancy_data(
                cache.get("raw_data_ref"), raw_data
            ) and cache.get("raw_data_key") is not None:
                # ROS deserialization normally allocates a fresh Python list
                # for every unchanged map frame.  A content comparison is
                # still O(N), but it runs in CPython's list implementation and
                # avoids allocating a NumPy buffer and hashing the complete
                # map on every publish tick.  The previous list is already
                # retained by ``raw_data_ref`` in the materialized cache.
                raw_data_key = cache.get("raw_data_key")
            else:
                raw_data_key = self._occupancy_data_key(raw)
            if (
                cache.get("mode") == "active"
                and cache.get("geometry_key") == geometry_key
                and cache.get("overlay_key") == overlay_key
                and cache.get("raw_data_key") == raw_data_key
            ):
                # Source headers advance with every mapper frame even when the
                # occupancy bytes are unchanged.  Retimestamp shallow ROS
                # messages while reusing the already materialized cell lists.
                cached_stats = copy.deepcopy(cache.get("overlay_stats") or {})
                return (
                    self._retime_cached_grid(cache.get("planning_data"), raw),
                    self._retime_cached_update(cache.get("planning_update"), raw),
                    self._retime_cached_grid(cache.get("mask_data"), raw),
                    cached_stats,
                )

            planning_data, mask_data, overlay_stats = self.semantic_occ_overlay.apply(
                raw.info,
                raw.data,
                include_pending=True,
            )
            planning_grid = self._retime_cached_grid(planning_data, raw)
            door_clear_mask = self._retime_cached_grid(mask_data, raw)
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
            self._planning_products_cache = {
                "mode": "active",
                "geometry_key": geometry_key,
                "overlay_key": overlay_key,
                "raw_data_key": raw_data_key,
                "raw_data_ref": raw_data,
                "planning_data": planning_data,
                "mask_data": mask_data,
                "planning_update": planning_update,
                "overlay_stats": copy.deepcopy(overlay_stats or {}),
            }
            return planning_grid, planning_update, door_clear_mask, overlay_stats

    def _build_planning_products_locked(self, raw, graph_payload):
        """Legacy synchronous implementation for tests that bypass ``__init__``."""

        # Lightweight compatibility doubles may provide only the pending-state
        # API used by the interaction callback.  In that case there is no
        # overlay product to materialize; forwarding the source-new raw grid
        # still preserves the causal planner barrier and is equivalent to an
        # inactive overlay in the production implementation.
        if not (
            callable(getattr(self.semantic_occ_overlay, "update_graph", None))
            and callable(getattr(self.semantic_occ_overlay, "apply", None))
        ):
            return raw, None, None, {
                "active_portal_ids": [],
                "cleared_cells": 0,
                "update_bounds": None,
                "valid": True,
            }
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
        graph_payload = self._materialize_graph_payload_from_snapshot(snapshot)
        causal_core_only = bool(snapshot.get("causal_core_only", False))
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
            "graph_cache_token": snapshot.get("graph_cache_token"),
            "publish_graph_products": bool(
                snapshot.get("publish_graph_products", True)
            ),
            # Keep the exact raw source in the emitted bundle.  Strict
            # readiness is evaluated after publishing and must never read a
            # newer ``latest_occupancy_grid`` by accident.
            "raw_occupancy_grid": snapshot.get("raw_occupancy_grid"),
            "room_attribute_request": (
                None
                if causal_core_only
                or not bool(snapshot.get("publish_graph_products", True))
                else self._build_room_attribute_request_locked(graph_payload)
            ),
            "planning_grid": planning_grid,
            "planning_update": planning_update,
            "door_clear_mask": door_clear_mask,
            "overlay_stats": overlay_stats,
            "causal_core_only": causal_core_only,
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
        publish_graph_products = bool(
            bundle.get("publish_graph_products", True)
        )
        graph_cache_token = bundle.get("graph_cache_token")
        room_attribute_request = bundle["room_attribute_request"]
        planning_grid = bundle["planning_grid"]
        planning_update = bundle["planning_update"]
        door_clear_mask = bundle["door_clear_mask"]
        causal_core_only = bool(bundle.get("causal_core_only", False))
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
            now = time.monotonic()
            if now - self._last_room_grid_publish_mono >= 1.0 / self.room_grid_publish_rate:
                self.room_segment_pub.publish(room_segment_grid)
                self._last_room_grid_publish_mono = now
        if planning_grid is not None:
            self.planning_occupancy_grid_pub.publish(planning_grid)
        if door_clear_mask is not None:
            self.door_clear_mask_pub.publish(door_clear_mask)
        if planning_update is not None:
            self.planning_occupancy_grid_updates_pub.publish(planning_update)
        graph_payload = {**graph_payload, "frame_id": self.world_frame}
        if publish_graph_products:
            self.unified_graph_pub.publish(String(data=dumps_compact(graph_payload)))
            if room_attribute_request is not None:
                self.room_attribute_requests_pub.publish(
                    String(data=dumps_compact(room_attribute_request))
                )
            self.navigation_hints_pub.publish(String(data=dumps_compact(graph_payload["views"]["navigation_view"]["hints"])))
            if not causal_core_only:
                self.unified_graph_markers_pub.publish(
                    build_graph_marker_array(graph_payload, self.world_frame)
                )
                self.unified_graph_markers_lifted_pub.publish(
                    build_graph_marker_array(
                        graph_payload,
                        self.lifted_graph_frame,
                    )
                )
                self._save_graph_payload(graph_payload)
            with self.lock:
                self._last_graph_publish_token = graph_cache_token
                self._last_graph_publish_mono = time.monotonic()

    @staticmethod
    def _same_occupancy_data(left, right):
        """Compare two ROS occupancy data sequences without NumPy allocation."""

        if left is right:
            return True
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        try:
            return len(left) == len(right) and left == right
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _occupancy_data_key(grid):
        """Return a compact identity for occupancy *content*, not its header.

        GMapping commonly emits a fresh ``OccupancyGrid`` object (and stamp)
        even when no cell changed.  Comparing object IDs therefore defeats
        caching.  A fixed-size digest keeps the key small while preserving
        exact signed-int8 cell semantics.
        """

        info = getattr(grid, "info", None)
        expected = 0
        if info is not None:
            try:
                expected = int(info.width) * int(info.height)
            except (TypeError, ValueError, AttributeError):
                expected = 0
        data = getattr(grid, "data", ())
        try:
            values = np.asarray(data, dtype=np.int8).reshape(-1)
            values = np.ascontiguousarray(values)
            digest = hashlib.blake2b(
                values.view(np.uint8), digest_size=16
            ).digest()
            return expected, int(values.size), digest
        except (TypeError, ValueError, OverflowError):
            # Malformed maps are handled by the existing overlay validity
            # path; retaining object identity here prevents a false cache hit.
            try:
                length = len(data)
            except TypeError:
                length = -1
            return expected, int(length), id(data)

    def _planning_overlay_key(self):
        """Capture every overlay field that can alter materialized cells."""

        overlay = self.semantic_occ_overlay
        references = []
        for node_id, reference in sorted(
            (getattr(overlay, "reference_aabbs", {}) or {}).items(),
            key=lambda item: str(item[0]),
        ):
            try:
                center, size = reference
                center_key = tuple(round(float(value), 9) for value in center[:3])
                size_key = tuple(round(float(value), 9) for value in size[:3])
            except (TypeError, ValueError, IndexError):
                center_key = ("invalid", repr(reference))
                size_key = ()
            references.append((str(node_id), center_key, size_key))
        return (
            bool(getattr(overlay, "enabled", False)),
            tuple(sorted(str(value) for value in (getattr(overlay, "active_portal_ids", None) or []))),
            tuple(references),
            round(float(getattr(overlay, "clear_padding_m", 0.0)), 9),
            round(float(getattr(overlay, "max_aperture_thickness_m", 0.0)), 9),
        )

    def _retime_cached_grid(self, data, raw):
        """Wrap cached cell storage with the latest source header/geometry.

        ``data`` is intentionally shared with the cache.  The ROS serializer
        reads it synchronously during ``publish`` and never mutates the list.
        """

        if data is None or raw is None:
            return None
        if isinstance(data, OccupancyGrid):
            data = data.data
        if not isinstance(data, list):
            data = list(data)
        grid = OccupancyGrid()
        raw_header = getattr(raw, "header", None)
        if raw_header is not None:
            grid.header.seq = getattr(raw_header, "seq", 0)
            grid.header.stamp = getattr(raw_header, "stamp", rospy.Time(0))
            grid.header.frame_id = getattr(raw_header, "frame_id", "") or getattr(
                self, "world_frame", ""
            )
        grid.info = raw.info
        grid.data = data
        return grid

    def _retime_cached_update(self, update, raw):
        """Retimestamp an incremental map update without copying its payload."""

        if update is None or raw is None:
            return None
        result = OccupancyGridUpdate()
        raw_header = getattr(raw, "header", None)
        if raw_header is not None:
            result.header.seq = getattr(raw_header, "seq", 0)
            result.header.stamp = getattr(raw_header, "stamp", rospy.Time(0))
            result.header.frame_id = getattr(raw_header, "frame_id", "") or getattr(
                self, "world_frame", ""
            )
        result.x = int(update.x)
        result.y = int(update.y)
        result.width = int(update.width)
        result.height = int(update.height)
        result.data = update.data if isinstance(update.data, list) else list(update.data)
        return result

    def _build_occupancy_copy(self, data, *, raw=None):
        if raw is None:
            raw = self.latest_occupancy_grid
        grid = NumpyOccupancyGrid()
        # Keep the raw map timestamp so downstream consumers can pair the
        # semantic overlay and clear mask with the exact source occupancy map.
        grid.header.seq = raw.header.seq
        grid.header.stamp = raw.header.stamp
        grid.header.frame_id = raw.header.frame_id or self.world_frame
        grid.info = raw.info
        grid.data = occupancy_data_snapshot(data)
        return grid

    def _build_room_segment_grid(self, room_ids, *, raw=None, encoded_data=None):
        if raw is None:
            raw = self.latest_occupancy_grid
        width = int(raw.info.width)
        height = int(raw.info.height)
        # Publish the room raster in exactly the raw occupancy geometry.  The
        # previous valid-ID crop made the room overlay look like a small dark
        # island and shifted it relative to OCC.  Cells outside discovered
        # free space remain -1 (unknown); they are intentionally not assigned
        # a fabricated room identity.
        row_min, row_max = 0, height
        col_min, col_max = 0, width

        def new_grid_shell():
            grid = OccupancyGrid()
            grid.header.seq = raw.header.seq
            grid.header.stamp = raw.header.stamp
            grid.header.frame_id = raw.header.frame_id or self.world_frame
            grid.info = copy.deepcopy(raw.info)
            grid.info.width = col_max - col_min
            grid.info.height = row_max - row_min

            origin = grid.info.origin
            yaw = math.atan2(
                2.0
                * (
                    origin.orientation.w * origin.orientation.z
                    + origin.orientation.x * origin.orientation.y
                ),
                1.0
                - 2.0
                * (
                    origin.orientation.y * origin.orientation.y
                    + origin.orientation.z * origin.orientation.z
                ),
            )
            resolution = float(grid.info.resolution)
            offset_x = float(col_min) * resolution
            offset_y = float(row_min) * resolution
            origin.position.x += math.cos(yaw) * offset_x - math.sin(yaw) * offset_y
            origin.position.y += math.sin(yaw) * offset_x + math.cos(yaw) * offset_y
            return grid

        # The cache key guarantees geometry and payload length.  Keep this
        # fast path before ``np.asarray(room_ids)`` so a header-only update
        # does not materialize the full room-ID raster again.
        if encoded_data is not None:
            try:
                if (
                    len(room_ids) == width * height
                    and len(encoded_data) == width * height
                ):
                    grid = new_grid_shell()
                    # The cached list is immutable by convention after it is
                    # inserted into the topology cache.  Reusing it avoids a
                    # second full-map allocation while preserving the exact
                    # ROS header/info behavior of the uncached path.
                    grid.data = encoded_data
                    return grid
            except (TypeError, ValueError):
                pass

        values = np.asarray(room_ids, dtype=np.int32)
        if values.size != width * height:
            raise ValueError("room grid size does not match occupancy geometry")
        cropped = values.reshape(height, width)
        grid = new_grid_shell()
        # ``nav_msgs/OccupancyGrid.data`` is signed int8.  Stable room IDs are
        # deliberately monotonically allocated and can exceed 127 during a
        # long, changing episode; publishing those internal IDs directly makes
        # rospy reject the *entire* room-grid message.  The ROS grid is only a
        # visualization/debug label raster (the graph keeps the full stable
        # IDs), so remap the currently visible positive labels to a compact
        # int8 palette only when necessary.  Keep the ordinary small-ID path
        # byte-for-byte compatible for consumers that inspect it.
        flat = cropped.reshape(-1)
        if np.any(flat > 127) or np.any(flat < -128):
            encoded = np.full(flat.shape, -1, dtype=np.int16)
            visible_ids = np.unique(flat[flat >= 0])
            for palette_index, stable_id in enumerate(visible_ids.tolist()):
                # A room-grid render can display at most 127 distinct positive
                # labels.  Collisions beyond that are visual-only; graph data
                # remains lossless and continues to use ``stable_id``.
                encoded[flat == stable_id] = 1 + (palette_index % 127)
            grid.data = encoded.tolist()
        else:
            grid.data = flat.astype(np.int16, copy=False).tolist()
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
            round(float(origin.position.z), 6),
            round(float(origin.orientation.x), 6),
            round(float(origin.orientation.y), 6),
            round(float(origin.orientation.z), 6),
            round(float(origin.orientation.w), 6),
            str(grid.header.frame_id or ""),
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
        supplied = secs is not None
        if supplied:
            try:
                stamp = float(secs) + float(nsecs or 0) * 1e-9
                if math.isfinite(stamp):
                    return stamp
            except (TypeError, ValueError, OverflowError):
                pass
        for key in ("stamp_sec", "stamp", "capture_stamp_sec"):
            if parsed.get(key) is None:
                continue
            supplied = True
            try:
                stamp = float(parsed[key])
                if math.isfinite(stamp):
                    return stamp
            except (TypeError, ValueError, OverflowError):
                pass
        # Invalid explicit capture metadata must not become a fresh receipt.
        return math.nan if supplied else rospy.Time.now().to_sec()

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
