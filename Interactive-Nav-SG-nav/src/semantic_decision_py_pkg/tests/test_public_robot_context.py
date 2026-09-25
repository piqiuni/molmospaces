import math

from semantic_decision_py_pkg.public_robot_context import graph_robot_pose_context


def test_nonidentity_tf_transforms_model_xy_and_records_original_pose():
    result = graph_robot_pose_context(
        [1, 2, 0], "odom", "map", source_stamp_sec=12,
        transform=([10, -3, 0], [0, 0, math.sqrt(0.5), math.sqrt(0.5)]),
    )
    assert abs(result["robot_graph_xy"][0] - 8) < 1e-6
    assert abs(result["robot_graph_xy"][1] + 2) < 1e-6
    assert result["robot_graph_frame_id"] == "map"
    assert result["robot_graph_pose_source"] == {
        "source_xy": [1, 2], "source_frame_id": "odom", "source_pose_stamp_sec": 12,
        "graph_frame_id": "map", "kind": "tf_transform",
    }


def test_missing_tf_does_not_assume_identity():
    result = graph_robot_pose_context([1, 2, 0], "odom", "map")
    assert result["robot_graph_xy"] is None
    assert result["robot_graph_pose_source"]["reason"] == "transform_unavailable"


def test_same_frame_does_not_need_tf_and_unknown_frame_is_not_map():
    same = graph_robot_pose_context([1, 2, 0], "/map", "map")
    assert same["robot_graph_xy"] == [1, 2]
    assert same["robot_graph_pose_source"]["kind"] == "same_frame"
    unknown = graph_robot_pose_context([1, 2, 0], "", "map")
    assert unknown["robot_graph_xy"] is None
    assert unknown["robot_graph_pose_source"]["reason"] == "coordinate_frame_unknown"
