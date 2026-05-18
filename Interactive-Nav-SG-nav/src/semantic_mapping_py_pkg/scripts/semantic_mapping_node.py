#!/usr/bin/env python3
import math
import threading

import rospy
import sensor_msgs.point_cloud2 as pc2
import tf
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from semantic_mapping_py_pkg.geometry_utils import normalize_label, transform_point
from semantic_mapping_py_pkg.messages import dumps_compact, parse_json_list, parse_json_object_or_text
from semantic_mapping_py_pkg.ros_params import get_frames, get_nested_param, get_topics
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore, SceneGridStore


class SemanticMappingNode:
    def __init__(self):
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

        self.object_map_topic = topics.get("object_map", "/semantic_mapping/obj_map")
        self.object_markers_topic = topics.get("object_markers", "/semantic_mapping/object_semantic_map_markers")
        self.scene_id_grid_topic = topics.get("scene_id_grid", "/semantic_mapping/scene_id_grid")
        self.scene_confidence_grid_topic = topics.get("scene_confidence_grid", "/semantic_mapping/scene_confidence_grid")

        self.enable_object_mapping = bool(config.get("enable_object_mapping", True))
        self.enable_scene_mapping = bool(config.get("enable_scene_mapping", True))
        self.publish_rate = float(config.get("publish_rate", 2.0))
        self.scene_min_range = float(config.get("scene_min_range", 0.1))
        self.scene_max_range = float(config.get("scene_max_range", 3.0))
        self.class_to_id = {
            normalize_label(name): int(value)
            for name, value in (scene_types.get("class_to_id", {}) or {}).items()
        }
        self.synonyms = {
            normalize_label(src): normalize_label(dst)
            for src, dst in (scene_types.get("synonyms", {}) or {}).items()
        }

        self.object_store = ObjectMapStore(
            match_distance=config.get("object_match_distance", 0.5),
            stale_after_sec=config.get("object_stale_after_sec", 0.0),
        )
        self.scene_store = SceneGridStore(
            unknown_id=scene_types.get("unknown_id", -1),
            confidence_step=config.get("scene_confidence_step", 5),
        )

        self.lock = threading.Lock()
        self.latest_cloud = None
        self.latest_scene = None
        self.tf_listener = tf.TransformListener()

        self.object_sub = rospy.Subscriber(self.object_detection_topic, String, self.object_callback, queue_size=10)
        self.scene_sub = rospy.Subscriber(self.scene_attribute_topic, String, self.scene_callback, queue_size=10)
        self.cloud_sub = rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.occ_sub = rospy.Subscriber(self.occupancy_grid_topic, OccupancyGrid, self.occupancy_callback, queue_size=1)

        self.object_pub = rospy.Publisher(self.object_map_topic, String, queue_size=1)
        self.marker_pub = rospy.Publisher(self.object_markers_topic, MarkerArray, queue_size=1)
        self.scene_id_pub = rospy.Publisher(self.scene_id_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.scene_conf_pub = rospy.Publisher(self.scene_confidence_grid_topic, OccupancyGrid, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.publish_callback)

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
        with self.lock:
            self.object_store.update(detections, msg._connection_header.get("stamp") if hasattr(msg, "_connection_header") else rospy.Time.now())

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

    def pointcloud_callback(self, msg):
        with self.lock:
            self.latest_cloud = msg

    def occupancy_callback(self, msg):
        with self.lock:
            self.scene_store.initialize_from_occupancy_grid(msg)

    def publish_callback(self, _event):
        if self.enable_object_mapping:
            obj_map = self.object_store.as_obj_map()
            self.object_pub.publish(String(data=dumps_compact(obj_map)))
            self.marker_pub.publish(self._build_object_markers(obj_map))

        if self.enable_scene_mapping and self.scene_store.info is not None:
            self.scene_id_pub.publish(self._build_grid(self.scene_store.scene_data))
            self.scene_conf_pub.publish(self._build_grid(self.scene_store.confidence_data))

    def _update_scene_from_cloud(self, cloud, scene_id):
        points = []
        try:
            self.tf_listener.waitForTransform(
                self.world_frame, cloud.header.frame_id, cloud.header.stamp, rospy.Duration(0.2)
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[semantic_mapping_node.py] TF unavailable for scene cloud: %s", exc)
            return

        for index, point in enumerate(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if index % 20 != 0:
                continue
            x, y, z = point
            dist = math.sqrt(x * x + y * y + z * z)
            if dist < self.scene_min_range or dist > self.scene_max_range:
                continue
            try:
                wx, wy, _wz = transform_point(
                    self.tf_listener, self.world_frame, cloud.header.frame_id, cloud.header.stamp, (x, y, z)
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
