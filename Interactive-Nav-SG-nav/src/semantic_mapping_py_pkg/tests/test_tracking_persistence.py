"""Tracking memory is independent from M1/action/topology admission."""

import copy

import pytest

from semantic_mapping_py_pkg.graph_rules import observation_from_detection
from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore


def detection(label="door"):
    return {
        "semantic_class": label,
        "confidence": 0.9,
        "world_position": [2.0, 0.0, 1.0],
        "world_box3d_center": [2.0, 0.0, 1.0],
        "world_box3d_size": [0.1, 0.9, 2.0],
    }


def stores(classes=("door", "portal", "locker", "fridge", "microwave"), require_m1=False):
    return (
        ObjectMapStore(stale_after_sec=30.0, min_confirmations=2,
                       persistent_classes=classes, persistence_requires_m1=require_m1),
        InteractionGraphStore(persistent_object_classes=classes,
                              object_persistence_requires_m1=require_m1),
    )


def capture(tracker, graph, detections, stamp):
    tracker.update(detections, stamp)
    exposed = tracker.as_tracked_detections(min_observations=2, confirmed_only=False,
                                           currently_observed_only=True)
    graph.update_observations([observation_from_detection(item, str(index))
                              for index, item in enumerate(exposed)],
                              stamp=stamp, source_mode="detector_online")
    graph.prune_stale_nodes(30.0, now=stamp)


@pytest.mark.parametrize("label", ["door", "locker", "fridge", "microwave"])
def test_two_source_frames_keep_identity_box_and_graph_without_m1(label):
    tracker, graph = stores()
    capture(tracker, graph, [detection(label)], 1.0)
    capture(tracker, graph, [detection(label)], 2.0)
    track_id = tracker.objects[0]["track_id"]
    node_id = next(key for key, node in graph.nodes.items() if node.type != "scene")
    box = copy.deepcopy(graph.nodes[node_id].aabb_size)
    capture(tracker, graph, [], 300.0)
    assert tracker.objects[0]["track_id"] == track_id
    node = graph.nodes[node_id]
    assert node.attributes["persistent_tracking_node"] is True
    assert node.aabb_size == box
    assert node.is_currently_visible is False
    assert not node.attributes.get("persistent_semantic_node")
    assert not any(node.type == "room" for node in graph.nodes.values())
    # One reappearance must not revoke prior temporal confirmation.
    capture(tracker, graph, [detection(label)], 301.0)
    assert tracker.objects[0]["track_id"] == track_id
    assert tracker.objects[0]["is_confirmed"]
    capture(tracker, graph, [], 1000.0)
    assert node_id in graph.nodes
    assert tracker.objects[0]["track_id"] == track_id


def test_single_frame_and_nonconsecutive_hits_still_expire():
    tracker, graph = stores()
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [], 2.0)
    capture(tracker, graph, [detection()], 3.0)
    capture(tracker, graph, [], 100.0)
    assert not tracker.objects
    assert all(node.type == "scene" for node in graph.nodes.values())


@pytest.mark.parametrize("classes,require_m1", [((), False), (("door", "portal"), True)])
def test_retention_configuration_can_disable_or_require_m1(classes, require_m1):
    tracker, graph = stores(classes, require_m1)
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [detection()], 2.0)
    capture(tracker, graph, [], 100.0)
    assert not tracker.objects
    assert all(node.type == "scene" for node in graph.nodes.values())


def test_default_policy_keeps_unconfirmed_m1_ttl_behavior():
    tracker = ObjectMapStore(stale_after_sec=30.0)
    graph = InteractionGraphStore()
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [detection()], 2.0)
    capture(tracker, graph, [], 100.0)
    assert not tracker.objects
    assert all(node.type == "scene" for node in graph.nodes.values())


def test_m1_correction_can_revoke_configured_track_memory():
    tracker, graph = stores()
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [detection()], 2.0)
    node_id = next(key for key, node in graph.nodes.items() if node.type != "scene")
    node = graph.nodes[node_id]
    graph.apply_attribute_patch({"object_id": node_id, "attribute_status": "ready",
                                 "source": "mllm_attribute_inference", "confidence": 0.9,
                                 "interaction_class": "object", "observed_object_name": "wall",
                                 "interactable": False, "observation_signature": "corrected"},
                                stamp=3.0)
    assert not node.attributes.get("persistent_tracking_node")
    assert not node.attributes.get("persistent_semantic_node")
    tracker.update_retention_evidence(tracker.objects[0]["track_id"], node.attributes)
    capture(tracker, graph, [], 100.0)
    assert not tracker.objects
    assert node_id not in graph.nodes


def test_invisible_persistent_node_is_present_in_graph_markers():
    import rospy
    from semantic_mapping_py_pkg.interaction_graph_viz import build_graph_marker_array

    tracker, graph = stores()
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [detection()], 2.0)
    capture(tracker, graph, [], 100.0)
    markers = build_graph_marker_array(graph.as_graph_dict(stamp=100.0), "map",
                                        stamp=rospy.Time(100))
    boxes = [marker for marker in markers.markers
             if marker.ns == "interaction_graph_nodes"]
    assert len(boxes) == 1
    assert boxes[0].lifetime.to_sec() == 0.0


def test_episode_reset_clears_tracking_memory():
    tracker, graph = stores()
    capture(tracker, graph, [detection()], 1.0)
    capture(tracker, graph, [detection()], 2.0)
    tracker.reset()
    assert not tracker.objects
    graph.reset(episode_id="new")
    assert all(node.type == "scene" for node in graph.nodes.values())
