#!/usr/bin/env python3
import json
import math
import os
import threading

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
from semantic_mapping_py_pkg.interaction_graph_viz import build_graph_marker_array
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.messages import dumps_compact, parse_json_list, parse_json_object_or_text
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter, RoomSegmentationState
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore, SceneGridStore
from semantic_mapping_py_pkg.semantic_occ_overlay import OverlayUpdateRegionTracker, SemanticOccupancyOverlay


class SemanticMappingNode:
    def __init__(self):
        patch_roslogging_findcaller_for_py311()
        rospy.init_node("semantic_mapping_py")
        topics = get_topics(rospy)
        frames = get_frames(rospy)
        config = get_nested_param(rospy, "semantic_map", {}) or {}
        scene_types = get_nested_param(rospy, "scene_types", {}) or {}

        self.world_frame = frames.get("world_frame", "tf_frame_map")
        self.object_detection_topic = topics.get("object_detections", "/semantic_mapping/object_detections")
        self.scene_attribute_topic = topics.get("scene_attribute", "/semantic_mapping/scene_attribute")
        self.pointcloud_topic = topics.get("pointcloud", "/registered_scan")
        self.occupancy_grid_topic = topics.get("occupancy_grid", "/struct_mapping/occ_map")
        self.room_context_topic = topics.get("room_context", "/semantic_mapping/room_context")
        self.gt_observations_topic = topics.get("gt_observations", "/semantic_mapping/gt_observations")
        self.interaction_result_topic = topics.get("interaction_result", "/semantic_mapping/interaction_result")
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
            room_id_to_name=self.id_to_class,
            room_box_height=self.room_box_height,
            portal_closed_threshold=graph_config.get("portal_closed_threshold", 0.10),
            portal_open_threshold=graph_config.get("portal_open_threshold", 0.67),
        )
        self.semantic_occ_overlay = SemanticOccupancyOverlay(
            enabled=overlay_config.get("enabled", True),
            clear_padding_m=overlay_config.get("clear_padding_m", 0.10),
            open_states=overlay_config.get("open_states", ["open"]),
        )
        self.semantic_occ_update_tracker = OverlayUpdateRegionTracker()

        self.lock = threading.Lock()
        self.latest_cloud = None
        self.latest_scene = None
        self.latest_occupancy_grid = None
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
            state=RoomSegmentationState(),
        )
        self.tf_listener = tf.TransformListener()

        self.object_sub = rospy.Subscriber(self.object_detection_topic, String, self.object_callback, queue_size=10)
        self.scene_sub = rospy.Subscriber(self.scene_attribute_topic, String, self.scene_callback, queue_size=10)
        self.cloud_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.occ_sub = rospy.Subscriber(self.occupancy_grid_topic, OccupancyGrid, self.occupancy_callback, queue_size=1)
        self.room_context_sub = rospy.Subscriber(self.room_context_topic, String, self.room_context_callback, queue_size=1)
        self.gt_observation_sub = rospy.Subscriber(
            self.gt_observations_topic, String, self.gt_observation_callback, queue_size=2
        )
        self.interaction_result_sub = rospy.Subscriber(
            self.interaction_result_topic, String, self.interaction_result_callback, queue_size=2
        )

        self.object_pub = rospy.Publisher(self.object_map_topic, String, queue_size=1)
        self.marker_pub = rospy.Publisher(self.object_markers_topic, MarkerArray, queue_size=1)
        self.scene_id_pub = rospy.Publisher(self.scene_id_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.scene_conf_pub = rospy.Publisher(self.scene_confidence_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.unified_graph_pub = rospy.Publisher(self.unified_graph_topic, String, queue_size=1, latch=True)
        self.navigation_hints_pub = rospy.Publisher(self.navigation_hints_topic, String, queue_size=1, latch=True)
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

        rospy.loginfo("[semantic_mapping_node.py] object_in=%s scene_in=%s cloud=%s occ=%s",
                      self.object_detection_topic, self.scene_attribute_topic,
                      self.pointcloud_topic, self.occupancy_grid_topic)

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
            portal_structure_changed = self.room_segmenter.update_portal_hints(
                observations,
                source_mode="detector_online",
            )
            self.graph_store.update_observations(observations, stamp=stamp, source_mode="detector_online")
            if portal_structure_changed:
                self._refresh_room_grid_locked()
            self.graph_store.prune_stale_nodes(self.object_stale_after_sec, now=stamp)
            publish_bundle = self._collect_publish_bundle_locked()
        self._safe_publish_bundle(publish_bundle)

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
        episode_id = str(parsed.get("episode_id") or "")
        stamp = float(parsed.get("stamp_sec", rospy.Time.now().to_sec()))
        with self.lock:
            episode_changed = episode_id and episode_id != self.graph_store.episode_id
            if bool(parsed.get("episode_reset")) or episode_changed:
                self._save_episode_graph_locked(final=True)
                self.graph_store.reset(episode_id=episode_id, source_mode="realtime_gt_observation")
                self.semantic_occ_overlay.reset()
                self.semantic_occ_update_tracker.reset()
                self.object_store.objects = []
                self.object_store.next_id = 1
                self.room_segmenter.state = RoomSegmentationState()
            portal_structure_changed = self.room_segmenter.update_portal_hints(
                observations,
                source_mode="realtime_gt_observation",
            )
            if bool(parsed.get("episode_reset")) or episode_changed or portal_structure_changed:
                self._refresh_room_grid_locked()
            self.graph_store.update_observations(
                observations,
                stamp=stamp,
                source_mode="realtime_gt_observation",
            )
            publish_bundle = self._collect_publish_bundle_locked()
        self._safe_publish_bundle(publish_bundle)

    def interaction_result_callback(self, msg):
        parsed = parse_json_object_or_text(msg.data)
        stamp = float(parsed.get("stamp_sec", rospy.Time.now().to_sec()))
        with self.lock:
            changed = self.graph_store.update_interaction_result(parsed, stamp=stamp)
            publish_bundle = self._collect_publish_bundle_locked() if changed else None
        if publish_bundle is not None:
            self._safe_publish_bundle(publish_bundle)

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
                for room_id, room_name in room_id_to_name.items():
                    try:
                        self.id_to_class[int(room_id)] = normalize_label(room_name)
                    except (TypeError, ValueError):
                        continue
            self.graph_store.set_room_geometries(rooms)

    def occupancy_callback(self, msg):
        with self.lock:
            self.latest_occupancy_grid = msg
            self.scene_store.initialize_from_occupancy_grid(msg)
            self._refresh_room_grid_locked()

    def _refresh_room_grid_locked(self):
        if self.latest_occupancy_grid is None:
            return
        room_ids, room_conf = self._segment_rooms_from_occupancy(self.latest_occupancy_grid)
        self.graph_store.update_room_grid(
            self.latest_occupancy_grid.info,
            room_ids,
            room_conf,
            room_id_to_name=self.id_to_class,
        )

    def publish_callback(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            publish_bundle = self._collect_publish_bundle_locked()
        self._safe_publish_bundle(publish_bundle)

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
        with self.lock:
            self.scene_store.update_cells(points, scene_id)

    def _build_grid(self, data):
        grid = OccupancyGrid()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = self.world_frame
        grid.info = self.scene_store.info
        grid.data = [int(v) for v in data]
        return grid

    def _collect_publish_bundle_locked(self):
        obj_map = self.object_store.as_obj_map() if self.enable_object_mapping else None
        scene_info_ready = self.enable_scene_mapping and self.scene_store.info is not None
        scene_grid = self._build_grid(self.scene_store.scene_data) if scene_info_ready else None
        scene_conf_grid = self._build_grid(self.scene_store.confidence_data) if scene_info_ready else None
        graph_payload = self.graph_store.as_graph_dict()
        self.semantic_occ_overlay.update_graph(graph_payload)
        planning_grid = None
        planning_update = None
        door_clear_mask = None
        overlay_stats = {
            "active_portal_ids": [],
            "cleared_cells": 0,
            "update_bounds": None,
            "valid": False,
        }
        if self.latest_occupancy_grid is not None:
            planning_data, mask_data, overlay_stats = self.semantic_occ_overlay.apply(
                self.latest_occupancy_grid.info,
                self.latest_occupancy_grid.data,
            )
            planning_grid = self._build_occupancy_copy(planning_data)
            door_clear_mask = self._build_occupancy_copy(mask_data)
            update_region = self.semantic_occ_update_tracker.build(
                self.latest_occupancy_grid.info.width,
                self.latest_occupancy_grid.info.height,
                planning_data,
                overlay_stats.get("update_bounds"),
                geometry_key=self._occupancy_geometry_key(self.latest_occupancy_grid),
            )
            if update_region is not None:
                planning_update = self._build_occupancy_update(planning_grid, update_region)
        return {
            "obj_map": obj_map,
            "scene_grid": scene_grid,
            "scene_conf_grid": scene_conf_grid,
            "graph_payload": graph_payload,
            "planning_grid": planning_grid,
            "planning_update": planning_update,
            "door_clear_mask": door_clear_mask,
            "overlay_stats": overlay_stats,
        }

    def _publish_bundle(self, bundle):
        obj_map = bundle["obj_map"]
        scene_grid = bundle["scene_grid"]
        scene_conf_grid = bundle["scene_conf_grid"]
        graph_payload = bundle["graph_payload"]
        planning_grid = bundle["planning_grid"]
        planning_update = bundle["planning_update"]
        door_clear_mask = bundle["door_clear_mask"]
        if obj_map is not None:
            self.object_pub.publish(String(data=dumps_compact(obj_map)))
            self.marker_pub.publish(self._build_object_markers(obj_map))
        if scene_grid is not None and scene_conf_grid is not None:
            self.scene_id_pub.publish(scene_grid)
            self.scene_conf_pub.publish(scene_conf_grid)
        if planning_grid is not None and door_clear_mask is not None:
            self.planning_occupancy_grid_pub.publish(planning_grid)
            self.door_clear_mask_pub.publish(door_clear_mask)
            if planning_update is not None:
                self.planning_occupancy_grid_updates_pub.publish(planning_update)
        self.unified_graph_pub.publish(String(data=dumps_compact(graph_payload)))
        self.navigation_hints_pub.publish(String(data=dumps_compact(graph_payload["views"]["navigation_view"]["hints"])))
        self.unified_graph_markers_pub.publish(build_graph_marker_array(graph_payload, self.world_frame))
        self.unified_graph_markers_lifted_pub.publish(
            build_graph_marker_array(
                graph_payload,
                self.lifted_graph_frame,
            )
        )
        self._save_graph_payload(graph_payload)

    def _build_occupancy_copy(self, data):
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

    def _publish_lifted_graph_tf(self):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = self.world_frame
        tf_msg.child_frame_id = self.lifted_graph_frame
        tf_msg.transform.translation.z = float(self.lifted_graph_z_offset)
        tf_msg.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(tf_msg)

    def _segment_rooms_from_occupancy(self, occ_grid):
        return self.room_segmenter.segment(occ_grid)

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
