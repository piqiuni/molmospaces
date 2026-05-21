from geometry_msgs.msg import Point
import rospy
from visualization_msgs.msg import Marker, MarkerArray


TYPE_TO_COLOR = {
    "room": (0.9, 0.9, 0.2, 0.18),
    "portal": (0.95, 0.3, 0.2, 0.9),
    "support": (0.1, 0.55, 0.95, 0.85),
    "container": (0.95, 0.55, 0.1, 0.85),
    "object": (0.2, 0.85, 0.45, 0.9),
}

RELATION_TO_COLOR = {
    "in_room": (0.8, 0.8, 0.8, 0.7),
    "has_child": (0.5, 0.85, 1.0, 0.75),
    "supports": (0.15, 0.75, 1.0, 0.9),
    "contains": (1.0, 0.7, 0.2, 0.9),
    "connects": (1.0, 0.35, 0.25, 0.95),
    "adjacent_via": (1.0, 0.2, 0.65, 0.9),
}


def build_graph_marker_array(graph_payload, frame_id, stamp=None):
    markers = MarkerArray()
    stamp = stamp or rospy.Time.now()
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.header.stamp = stamp
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    nodes = {node["id"]: node for node in graph_payload.get("nodes", [])}
    edges = list(graph_payload.get("edges", []))
    supported_object_ids = {
        edge["dst_id"]
        for edge in edges
        if edge.get("relation") == "supports"
    }
    connect_pairs = {
        frozenset((edge["src_id"], edge["dst_id"]))
        for edge in edges
        if edge.get("relation") == "connects"
    }

    for index, node in enumerate(graph_payload.get("nodes", [])):
        color = TYPE_TO_COLOR.get(node["type"], (1.0, 1.0, 1.0, 0.9))
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "interaction_graph_nodes"
        marker.id = index * 3
        marker.action = Marker.ADD
        marker.type = Marker.CUBE
        marker.pose.position.x = float(node["aabb_center"][0])
        marker.pose.position.y = float(node["aabb_center"][1])
        marker.pose.position.z = float(node["aabb_center"][2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(float(node["aabb_size"][0]), 0.15)
        marker.scale.y = max(float(node["aabb_size"][1]), 0.15)
        marker.scale.z = max(float(node["aabb_size"][2]), 0.15)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        markers.markers.append(marker)

        label = Marker()
        label.header = marker.header
        label.ns = "interaction_graph_labels"
        label.id = index * 3 + 1
        label.action = Marker.ADD
        label.type = Marker.TEXT_VIEW_FACING
        label.pose.position.x = float(node["aabb_center"][0])
        label.pose.position.y = float(node["aabb_center"][1])
        label.pose.position.z = float(node["aabb_center"][2]) + max(float(node["aabb_size"][2]), 0.2) * 0.6 + 0.1
        label.pose.orientation.w = 1.0
        label.scale.z = 0.2
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"{node['type']}:{node['label']}"
        markers.markers.append(label)

    edge_offset = len(graph_payload.get("nodes", [])) * 3
    marker_index = 0
    for edge in edges:
        src = nodes.get(edge["src_id"])
        dst = nodes.get(edge["dst_id"])
        if src is None or dst is None:
            continue
        relation = edge.get("relation")
        edge_pair = frozenset((edge["src_id"], edge["dst_id"]))
        if relation == "adjacent_via" and edge_pair in connect_pairs:
            continue
        if relation in {"in_room", "has_child"}:
            object_node = (
                src
                if src.get("type") == "object"
                else dst
                if dst.get("type") == "object"
                else None
            )
            room_node = src if src.get("type") == "room" else dst if dst.get("type") == "room" else None
            if (
                object_node is not None
                and room_node is not None
                and (
                    object_node["id"] in supported_object_ids
                    or object_node.get("parent_id") is not None
                )
            ):
                continue
        line = Marker()
        line.header.frame_id = frame_id
        line.header.stamp = stamp
        line.ns = "interaction_graph_edges"
        line.id = edge_offset + marker_index
        line.action = Marker.ADD
        line.type = Marker.LINE_LIST
        line.scale.x = 0.03
        color = RELATION_TO_COLOR.get(relation, (0.9, 0.9, 0.9, 0.75))
        line.color.r, line.color.g, line.color.b, line.color.a = color
        line.points = []
        for point in (src["centroid"], dst["centroid"]):
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2])
            line.points.append(p)
        markers.markers.append(line)
        marker_index += 1
    return markers
