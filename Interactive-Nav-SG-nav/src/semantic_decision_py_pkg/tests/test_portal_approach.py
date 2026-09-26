import copy
import math
from dataclasses import asdict

import pytest

from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator, CandidateGeneratorConfig
from semantic_decision_py_pkg.behavior_execution import candidate_with_effective_interaction_approach
from semantic_decision_py_pkg.portal_approach import bounded_portal_tolerances, portal_approach_profile


def portal_candidate():
    return {
        "behavior_type": "INTERACT", "goal_xyyaw": [1.0, 0.0, math.pi],
        "interaction_command": {
            "node_type": "portal", "interaction_target_center_xy": [0.0, 0.0],
            "interaction_approach_axis_xy": [1.0, 0.0],
            "interaction_approach_pose_xyyaw": [1.0, 0.0, math.pi],
            "navigation_goal_position_tolerance_m": .15,
            "navigation_goal_yaw_tolerance_rad": .2,
            "interaction_front_position_tolerance_rad": .2,
            "interaction_front_yaw_tolerance_rad": .2,
        },
        "metadata": {"frame_id": "map", "portal_clearance_aware_approach": True,
                     "portal_approach_base_tolerances": [.15, .2]},
    }


def test_physical_portal_arrival_is_point_three_without_changing_default():
    candidate = portal_candidate()
    candidate["metadata"]["portal_approach_base_tolerances"] = [.3, .2]
    assert portal_approach_profile(candidate, [1., 0., math.pi])["distance_tolerance_m"] < .3
    candidate["metadata"]["portal_shrink_arrival_disc"] = False
    assert portal_approach_profile(candidate, [1., 0., math.pi])["distance_tolerance_m"] == .3
    assert portal_approach_profile(candidate, [-1., 0., 0.]) is None


@pytest.mark.parametrize("offset", [-.15, -.10, 0.0, .10, .15])
def test_portal_profile_reserves_the_whole_arrival_disc_and_yaw_interval(offset):
    goal = [1.5, offset, math.atan2(-offset, -1.5)]
    profile = bounded_portal_tolerances(goal, [0, 0], [1, 0], .15, .2, .2, .2)
    assert profile is not None
    for i in range(180):
        angle = i*2*math.pi/180
        x = goal[0] + profile["distance_tolerance_m"]*math.cos(angle)
        y = goal[1] + profile["distance_tolerance_m"]*math.sin(angle)
        assert abs(math.atan2(y, x)) < .2
    assert profile["planned_face_yaw_offset_rad"] + profile["yaw_tolerance_rad"] < .2


@pytest.mark.parametrize("goal", [
    [-1., 0., 0.], [0., 0., 0.], [1., .3, math.pi],
    [1., 0., 0.], [math.nan, 0., math.pi], [math.inf, 0., math.pi],
])
def test_portal_profile_rejects_wrong_face_or_invalid_goals(goal):
    assert portal_approach_profile(portal_candidate(), goal) is None


def test_selected_portal_contract_shrinks_without_redefining_face_or_primary():
    candidate = portal_candidate()
    before = copy.deepcopy(candidate)
    goal = [1., -.1, math.atan2(.1, -1.)]
    bound = candidate_with_effective_interaction_approach(candidate, goal, goal_option_index=1)
    command = bound["interaction_command"]
    assert command["interaction_approach_axis_xy"] == [1., 0.]
    assert command["navigation_goal_position_tolerance_m"] < .15
    assert command["navigation_goal_yaw_tolerance_rad"] < .2
    assert command["navigation_goal_position_tolerance_m"] == command["interaction_ready_distance_m"]
    assert command["navigation_goal_yaw_tolerance_rad"] == command["interaction_ready_yaw_tolerance_rad"]
    assert bound["goal_xyyaw"] == candidate["goal_xyyaw"]
    assert candidate == before
    normal = candidate_with_effective_interaction_approach(bound, candidate["goal_xyyaw"], goal_option_index=0)
    assert normal["interaction_command"]["navigation_goal_position_tolerance_m"] == .15


def test_remembered_h2_portal_advertises_bounded_same_face_tangents():
    generator = CandidateGenerator(CandidateGeneratorConfig(
        interaction_types=("portal",), portal_standoff_m=.85,
        portal_require_current_visibility=True, remembered_portal_reobservation_enabled=True,
        interaction_ready_yaw_tolerance_rad=.2,
    ))
    node = {
        "id": "door_0001", "type": "portal", "aabb_center": [3.24202, 4.2851365, 1.],
        "aabb_size": [.2199488, .9984034, 2.], "is_currently_visible": False,
        "interaction": {"is_interactable": True, "requires_interaction": True, "state": "unknown"},
    }
    candidate = generator.generate({}, {"nodes": [node]}, robot_xy=(2.62, .9))[0]
    assert candidate.candidate_id == "reobserve_portal:door_0001"
    data = asdict(candidate)
    goals = candidate.metadata["goal_xyyaw_candidates"]
    assert any(goal[1] < 4.2851365 for goal in goals)
    assert any(goal[1] > 4.2851365 for goal in goals)
    assert all(goal[0] < 3.24202 for goal in goals)
    assert all(portal_approach_profile(data, goal) is not None for goal in goals)
    assert len(goals) <= 15
