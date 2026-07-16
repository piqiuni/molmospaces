from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.portal_state_tracker import PortalStateTracker
from semantic_mapping_py_pkg.gt_observation_provider import observation_from_gt_record


def joint(name, value, limits):
    return {
        "joint_name": name,
        "joint_type": "hinge",
        "joint_range": list(limits),
        "joint_value": float(value),
    }


def test_positive_and_negative_door_ranges_use_episode_closed_reference():
    tracker = PortalStateTracker()
    positive_closed = tracker.update("positive", {"joint_infos": [joint("hinge", 0.0, [0.0, 1.0])]})
    negative_closed = tracker.update("negative", {"joint_infos": [joint("hinge", 0.0, [-1.0, 0.0])]})
    assert positive_closed["state"] == "closed"
    assert negative_closed["state"] == "closed"

    positive_open = tracker.update("positive", {"joint_infos": [joint("hinge", 0.8, [0.0, 1.0])]})
    negative_open = tracker.update("negative", {"joint_infos": [joint("hinge", -0.8, [-1.0, 0.0])]})
    assert positive_open["state"] == "open"
    assert negative_open["state"] == "open"
    assert positive_open["open_fraction"] == 0.8
    assert negative_open["open_fraction"] == 0.8


def test_double_door_requires_all_non_handle_hinges_to_open():
    tracker = PortalStateTracker()
    closed = {
        "joint_infos": [
            joint("left_hinge", 0.0, [0.0, 1.0]),
            joint("right_hinge", 0.0, [-1.0, 0.0]),
            joint("left_handle_hinge", 0.0, [0.0, 0.2]),
        ]
    }
    assert tracker.update("double", closed)["state"] == "closed"

    one_leaf_open = {
        "joint_infos": [
            joint("left_hinge", 0.9, [0.0, 1.0]),
            joint("right_hinge", 0.0, [-1.0, 0.0]),
            joint("left_handle_hinge", 0.2, [0.0, 0.2]),
        ]
    }
    assert tracker.update("double", one_leaf_open)["state"] == "ajar"

    both_open = {
        "joint_infos": [
            joint("left_hinge", 0.9, [0.0, 1.0]),
            joint("right_hinge", -0.9, [-1.0, 0.0]),
        ]
    }
    result = tracker.update("double", both_open)
    assert result["state"] == "open"
    assert set(result["joint_open_fractions"]) == {"left_hinge", "right_hinge"}


def test_partial_opening_is_ajar():
    tracker = PortalStateTracker(closed_threshold=0.1, open_threshold=0.67)
    tracker.update("door", {"joint_infos": [joint("hinge", 0.0, [0.0, 1.0])]})
    result = tracker.update("door", {"joint_infos": [joint("hinge", 0.4, [0.0, 1.0])]})
    assert result["state"] == "ajar"
    assert result["open_fraction"] == 0.4


def test_gt_replay_observation_preserves_all_door_joint_readbacks():
    record = {
        "name": "double_door_root",
        "category": "Doorway",
        "is_door": True,
        "is_movable_door": True,
        "is_articulable": True,
        "joint_infos": [
            joint("left_hinge", 0.0, [0.0, 1.0]),
            joint("right_hinge", 0.0, [-1.0, 0.0]),
        ],
    }
    observation = observation_from_gt_record(record, "door_obs")
    assert [info["joint_name"] for info in observation["joint_infos"]] == ["left_hinge", "right_hinge"]
    assert observation["primary_joint_name"] == "left_hinge"
