from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_mapping_py_pkg.portal_state_consensus import PortalStateConsensus


def test_initial_state_needs_three_distinct_consecutive_views_then_cools_down():
    consensus = PortalStateConsensus(
        confirmation_count=3,
        cooldown_steps=300,
        min_position_gap_m=0.25,
        min_yaw_gap_rad=0.25,
    )

    first = consensus.observe(
        "door_1", "closed", capture_step=10, observation_pose_xyyaw=[0.0, 0.0, 0.0]
    )
    second = consensus.observe(
        "door_1", "closed", capture_step=20, observation_pose_xyyaw=[0.4, 0.0, 0.0]
    )
    third = consensus.observe(
        "door_1", "closed", capture_step=30, observation_pose_xyyaw=[0.8, 0.0, 0.0]
    )

    assert first["accepted"] is False
    assert second["confirmation_count"] == 2
    assert third["accepted"] is True
    assert third["stable_state"] == "closed"
    assert third["reason"] == "initial_state_confirmed"
    assert consensus.can_request(
        "door_1", capture_step=329, observation_pose_xyyaw=[1.2, 0.0, 0.0]
    ) == (False, "stable_state_cooldown")
    assert consensus.can_request(
        "door_1", capture_step=330, observation_pose_xyyaw=[1.2, 0.0, 0.0]
    )[0]

    cached = consensus.cached_stable_result("door_1", capture_step=329)
    assert cached is not None
    assert cached["accepted"] is True
    assert cached["stable_state"] == "closed"
    assert cached["reason"] == "cached_stable_state_cooldown"
    assert consensus.cached_stable_result("door_1", capture_step=330) is None


def test_state_change_needs_three_distinct_consecutive_views():
    consensus = PortalStateConsensus(confirmation_count=3, cooldown_steps=10)
    for step, pose in ((1, [0.0, 0.0, 0.0]), (2, [0.3, 0.0, 0.0]), (3, [0.6, 0.0, 0.0])):
        consensus.observe("door_1", "closed", capture_step=step, observation_pose_xyyaw=pose)

    first_change = consensus.observe(
        "door_1", "open", capture_step=13, observation_pose_xyyaw=[1.0, 0.0, 0.0]
    )
    interrupted = consensus.observe(
        "door_1", "closed", capture_step=14, observation_pose_xyyaw=[1.3, 0.0, 0.0]
    )
    assert first_change["accepted"] is False
    assert interrupted["accepted"] is True
    assert interrupted["reason"] == "stable_state_reconfirmed"

    for step, pose in ((24, [1.6, 0.0, 0.0]), (25, [1.9, 0.0, 0.0])):
        result = consensus.observe(
            "door_1", "open", capture_step=step, observation_pose_xyyaw=pose
        )
        assert result["accepted"] is False
    changed = consensus.observe(
        "door_1", "open", capture_step=26, observation_pose_xyyaw=[2.2, 0.0, 0.0]
    )
    assert changed["accepted"] is True
    assert changed["stable_state"] == "open"
    assert changed["reason"] == "state_change_confirmed"


def test_duplicate_pose_does_not_advance_confirmation():
    consensus = PortalStateConsensus(confirmation_count=3)
    consensus.observe(
        "door_1", "closed", capture_step=1, observation_pose_xyyaw=[0.0, 0.0, 0.0]
    )
    allowed, reason = consensus.can_request(
        "door_1", capture_step=2, observation_pose_xyyaw=[0.1, 0.0, 0.1]
    )
    assert allowed is False
    assert reason == "duplicate_view_pose"


def test_authoritative_interaction_result_starts_cooldown_without_more_views():
    consensus = PortalStateConsensus(confirmation_count=3, cooldown_steps=300)

    assert consensus.record_authoritative(
        "door_1", "open", capture_step=100
    )
    assert consensus.can_request(
        "door_1", capture_step=399, observation_pose_xyyaw=[1.0, 2.0, 0.0]
    ) == (False, "stable_state_cooldown")
    assert consensus.can_request(
        "door_1", capture_step=400, observation_pose_xyyaw=[1.0, 2.0, 0.0]
    )[0]
