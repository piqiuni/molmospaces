#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from semantic_mapping_py_pkg.gt_observation_provider import observation_from_gt_record, split_observations_into_batches
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.interaction_graph_viz import build_graph_marker_array
from semantic_mapping_py_pkg.messages import dumps_compact
from semantic_mapping_py_pkg.ros_py311_compat import patch_roslogging_findcaller_for_py311


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish GT scene JSON as incrementally replayed detection messages."
    )
    parser.add_argument("input", type=Path, help="Path to full scene JSON exported by read_scene_room_properties.py.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of detections per published batch.")
    parser.add_argument("--publish-rate", type=float, default=1.0, help="Replay publish rate in Hz.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle records before batching.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when shuffling.")
    parser.add_argument("--loop", action="store_true", help="Loop over the scene forever.")
    parser.add_argument(
        "--topic",
        type=str,
        default="/semantic_mapping/object_detections",
        help="Detection topic to publish.",
    )
    parser.add_argument(
        "--room-context-topic",
        type=str,
        default="/semantic_mapping/room_context",
        help="Room context topic to publish.",
    )
    parser.add_argument(
        "--graph-topic",
        type=str,
        default="/semantic_mapping/unified_graph",
        help="Unified graph JSON topic to publish.",
    )
    parser.add_argument(
        "--hints-topic",
        type=str,
        default="/semantic_mapping/navigation_hints",
        help="Navigation hints topic to publish.",
    )
    parser.add_argument(
        "--markers-topic",
        type=str,
        default="/semantic_mapping/unified_graph_markers",
        help="RViz marker topic to publish.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="tf_frame_map",
        help="Frame id used for graph markers.",
    )
    return parser.parse_args()


def detection_from_observation(observation):
    position = observation.get("position") or [0.0, 0.0, 0.0]
    aabb_center = observation.get("aabb_center") or position
    aabb_size = observation.get("aabb_size") or [0.0, 0.0, 0.0]
    return {
        "instance_id": observation.get("instance_id") or "",
        "semantic_class": observation.get("semantic_name") or "object",
        "semantic_name": observation.get("semantic_name") or "object",
        "category": observation.get("category") or observation.get("semantic_name") or "object",
        "confidence": float(observation.get("confidence", 1.0)),
        "world_position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "box3d_center": {"x": float(aabb_center[0]), "y": float(aabb_center[1]), "z": float(aabb_center[2])},
        "box3d_size": {"x": float(aabb_size[0]), "y": float(aabb_size[1]), "z": float(aabb_size[2])},
        "size": {"x": float(aabb_size[0]), "y": float(aabb_size[1]), "z": float(aabb_size[2])},
        "room_id": observation.get("room_id"),
        "connected_room_ids": list(observation.get("connected_room_ids") or []),
        "parent": observation.get("parent"),
        "children": list(observation.get("children") or []),
        "is_receptacle": bool(observation.get("is_receptacle", False)),
        "is_pickup_candidate": bool(observation.get("is_pickup_candidate", False)),
        "is_articulable": bool(observation.get("is_articulable", False)),
        "is_door": bool(observation.get("is_door", False)),
        "is_movable_door": bool(observation.get("is_movable_door", False)),
        "joint_type": observation.get("joint_type", "none"),
        "joint_range": list(observation.get("joint_range") or [0.0, 0.0]),
        "joint_value": observation.get("joint_value"),
        "name": observation.get("name"),
        "asset_id": observation.get("asset_id"),
        "object_id": observation.get("object_id"),
        "source": observation.get("source", "gt_replay"),
    }


def build_batches(scene_json, batch_size, shuffle, seed):
    records = list(scene_json.get("records") or [])
    observations = [
        observation_from_gt_record(record, observation_id=f"gt_obs_{index:04d}")
        for index, record in enumerate(records, start=1)
    ]
    return split_observations_into_batches(
        observations,
        shuffle=shuffle,
        seed=seed,
        batch_size=batch_size,
    )


def build_room_context_payload(scene_json):
    return {
        "scene_id": scene_json.get("scene_id", "scene"),
        "room_id_to_name": dict(scene_json.get("room_id_to_name") or {}),
        "rooms": list(scene_json.get("rooms") or []),
        "room_to_objects": dict(scene_json.get("room_to_objects") or {}),
    }


def main():
    args = parse_args()
    scene_json = json.loads(args.input.read_text())
    patch_roslogging_findcaller_for_py311()
    rospy.init_node("semantic_mapping_gt_replay")
    detection_publisher = rospy.Publisher(args.topic, String, queue_size=10, latch=True)
    room_context_publisher = rospy.Publisher(args.room_context_topic, String, queue_size=1, latch=True)
    graph_publisher = rospy.Publisher(args.graph_topic, String, queue_size=1, latch=True)
    hints_publisher = rospy.Publisher(args.hints_topic, String, queue_size=1, latch=True)
    markers_publisher = rospy.Publisher(args.markers_topic, MarkerArray, queue_size=1, latch=True)
    batches = build_batches(scene_json, args.batch_size, args.shuffle, args.seed)
    rate = rospy.Rate(max(args.publish_rate, 1e-3))
    room_context_payload = build_room_context_payload(scene_json)
    room_context_publisher.publish(String(data=dumps_compact(room_context_payload)))
    room_id_to_name = {
        int(room_id): str(name)
        for room_id, name in (scene_json.get("room_id_to_name") or {}).items()
    }
    graph_store = InteractionGraphStore(
        scene_id=scene_json.get("scene_id", "scene"),
        room_id_to_name=room_id_to_name,
    )
    graph_store.set_room_geometries(scene_json.get("rooms") or [])

    rospy.loginfo(
        "[semantic_mapping_gt_replay] input=%s records=%d batches=%d topic=%s graph_topic=%s markers_topic=%s",
        args.input,
        len(scene_json.get("records") or []),
        len(batches),
        args.topic,
        args.graph_topic,
        args.markers_topic,
    )

    while not rospy.is_shutdown():
        graph_store = InteractionGraphStore(
            scene_id=scene_json.get("scene_id", "scene"),
            room_id_to_name=room_id_to_name,
        )
        graph_store.set_room_geometries(scene_json.get("rooms") or [])
        for batch_index, batch in enumerate(batches, start=1):
            stamp = rospy.Time.now()
            detections = [detection_from_observation(observation) for observation in batch]
            payload = {
                "scene_id": scene_json.get("scene_id", "scene"),
                "batch_index": batch_index,
                "stamp_sec": int(stamp.secs),
                "stamp_nsec": int(stamp.nsecs),
                "detections": detections,
            }
            detection_publisher.publish(String(data=dumps_compact(payload)))

            graph_store.update_observations(batch, source_mode="gt_replay")
            graph_payload = graph_store.as_graph_dict()
            graph_publisher.publish(String(data=dumps_compact(graph_payload)))
            hints_publisher.publish(String(data=dumps_compact(graph_payload["views"]["navigation_view"]["hints"])))
            markers_publisher.publish(build_graph_marker_array(graph_payload, args.frame_id, stamp=stamp))

            rospy.loginfo(
                "[semantic_mapping_gt_replay] published batch %d/%d (%d detections, %d graph nodes)",
                batch_index,
                len(batches),
                len(detections),
                len(graph_payload.get("nodes", [])),
            )
            rate.sleep()
            if rospy.is_shutdown():
                break
        if not args.loop:
            break


if __name__ == "__main__":
    main()
