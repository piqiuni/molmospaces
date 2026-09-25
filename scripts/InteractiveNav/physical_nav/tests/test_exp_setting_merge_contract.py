"""Physical deployment contracts at the exp-setting integration boundary."""

import math
from pathlib import Path

import pytest
import yaml

from physical_yoloe_bridge import _capture_observation_metadata
from semantic_mapping_py_pkg.image_frame_pairing import select_image_record
from semantic_mapping_py_pkg.portal_state_consensus import PortalStateConsensus


PHYSICAL = Path(__file__).resolve().parents[1]


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        assert key not in result, f"Duplicate configuration key: {key}"
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def test_physical_config_preserves_safety_and_uses_time_not_camera_frames():
    with (PHYSICAL / "config/semantic_shadow_override.yaml").open() as source:
        override = yaml.load(source, Loader=UniqueKeyLoader)
    with (PHYSICAL / "config/physical_nav.yaml").open() as source:
        config = yaml.load(source, Loader=UniqueKeyLoader)
    assert override["ablation"]["module3"] == "external_mllm_verified"
    assert override["candidate"]["progress_clock"] == "monotonic"
    assert override["candidate"]["portal_require_reference_yaw"]
    assert override["candidate"]["portal_approach_tangent_offsets_m"] == [0.0]
    assert not override["executor"]["rear_goal_prerotate_step_sync_enabled"]
    assert not override["executor"]["navigation_failure_recovery_enabled"]
    assert override["executor"]["post_interaction_costmap_fast_path_enabled"]
    assert not override["model"]["force_unentered_room_exploration"]
    assert config["attribute_inference"]["include_detector_class_hypothesis"]
    assert config["attribute_inference"]["portal_state_cooldown_s"] == 120.0
    assert config["semantic_map"]["object_min_confirmations"] == 2


def test_navigation_clock_advances_with_time_not_camera_receipts(monkeypatch):
    pytest.importorskip("rospy")
    import semantic_candidate_node as module

    node = object.__new__(module.SemanticCandidateNode)
    node.progress_clock = "monotonic"
    node.progress_clock_period_s = 0.2
    node.graph = {"capture_step": 10}
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    first = node._progress_clock_context()
    node.graph["capture_step"] = 10000
    assert node._progress_clock_context()["observation_step"] == first["observation_step"]
    now[0] += 4.0
    later = node._progress_clock_context()
    assert later["observation_step"] - first["observation_step"] == 20
    assert later["source_capture_step"] == 10000
    node.progress_clock = "capture_step"
    assert node._progress_clock_context()["observation_step"] == 10000


def test_portal_cooldown_is_wall_time_even_when_camera_stalls_or_restarts(monkeypatch):
    import semantic_mapping_py_pkg.portal_state_consensus as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    consensus = PortalStateConsensus(confirmation_count=1, cooldown_s=120.0)
    result = consensus.observe("door", "closed", capture_step=100, observation_pose_xyyaw=[1, 2, 0])
    assert result["accepted"] and result["cooldown_clock"] == "monotonic"
    for capture in (0, 100, 1000000):
        assert consensus.can_request("door", capture_step=capture, observation_pose_xyyaw=[1, 2, 0]) == (
            False, "stable_state_cooldown")
    now[0] = 221.0
    assert consensus.can_request("door", capture_step=100, observation_pose_xyyaw=[1, 2, 0])[0]
    # Physical M3 does not carry simulator steps; it still latches authoritative state.
    assert consensus.record_authoritative("door", "open", capture_step=None)
    assert consensus.cached_stable_result("door", capture_step=None)["stable_state"] == "open"


def test_capture_pose_and_image_identity_survive_yolo_to_m1():
    raw = {"seq": 45, "stamp": 1788583565.123456,
           "telemetry": {"position": [3., 4., .305], "yaw": math.pi / 2}}
    metadata = _capture_observation_metadata(raw, (720, 1280, 3))
    assert metadata["image_size"] == [1280, 720]
    assert metadata["observation_pose_xyyaw"] == pytest.approx([3, 4, math.pi / 2])
    # Topic-local Header.seq is not a capture step or source image sequence.
    sec, nsec = divmod(round(raw["stamp"] * 1e9), 10**9)
    record = {"stamp_key": (sec, nsec), "image_sequence": 7, "header_seq": 888}
    assert select_image_record(metadata, [record]) == (record, "matched")
    assert select_image_record({**metadata, "stamp_sec": raw["stamp"] + .1}, [record])[0] is None


def test_missing_capture_pose_is_not_fabricated_for_m1():
    result = _capture_observation_metadata({"seq": 1, "stamp": 10., "telemetry": {}}, (480, 848, 3))
    assert "observation_pose_xyyaw" not in result
