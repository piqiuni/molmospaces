def get_nested_param(rospy, path, default=None):
    """Read a slash-separated parameter from the node namespace, then global namespace."""
    private_name = "~" + path
    if rospy.has_param(private_name):
        return rospy.get_param(private_name, default)
    if rospy.has_param(path):
        return rospy.get_param(path, default)
    return default


def get_topics(rospy):
    return get_nested_param(rospy, "topics", {}) or {}


def get_frames(rospy):
    return get_nested_param(rospy, "frames", {}) or {}
