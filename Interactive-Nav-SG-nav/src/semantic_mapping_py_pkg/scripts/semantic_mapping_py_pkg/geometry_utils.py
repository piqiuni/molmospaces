import math


def point_dict(x=0.0, y=0.0, z=0.0):
    return {"x": float(x), "y": float(y), "z": float(z)}


def euclidean_2d(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def world_to_grid(x, y, grid_info):
    mx = int((float(x) - grid_info.origin.position.x) / grid_info.resolution)
    my = int((float(y) - grid_info.origin.position.y) / grid_info.resolution)
    if mx < 0 or my < 0 or mx >= grid_info.width or my >= grid_info.height:
        return None
    return mx, my


def grid_index(mx, my, width):
    return int(my) * int(width) + int(mx)


def normalize_label(label):
    return str(label or "").strip().lower().replace(" ", "_")


def transform_point(tf_listener, target_frame, source_frame, stamp, point):
    from geometry_msgs.msg import PointStamped

    msg = PointStamped()
    msg.header.frame_id = source_frame
    msg.header.stamp = stamp
    msg.point.x = float(point[0])
    msg.point.y = float(point[1])
    msg.point.z = float(point[2])
    out = tf_listener.transformPoint(target_frame, msg)
    return out.point.x, out.point.y, out.point.z


def transform_point_best_effort(tf_listener, target_frame, source_frame, stamp, point):
    import rospy

    last_exc = None
    for candidate_stamp in [stamp, rospy.Time(0)]:
        if candidate_stamp is None:
            continue
        try:
            return transform_point(tf_listener, target_frame, source_frame, candidate_stamp, point), candidate_stamp
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unable to transform point")
