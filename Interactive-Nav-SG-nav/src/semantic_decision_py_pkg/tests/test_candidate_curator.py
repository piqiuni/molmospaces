from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
from semantic_decision_py_pkg.candidate_curator import (
    CandidateCurator,
    CandidateCuratorConfig,
    candidate_history_key,
    validate_candidate_update,
)


def make_candidate(
    candidate_id: str,
    behavior_type: str,
    *,
    x: float,
    y: float = 0.0,
    room_id: int = 1,
    target_goal: bool = False,
    node_type: str = "",
    state: str = "closed",
    gain: float = 0.5,
) -> BehaviorCandidate:
    interaction = (
        {"action": "open", "interaction_group_id": "all_joints"}
        if behavior_type == "INTERACT"
        else {}
    )
    return BehaviorCandidate(
        candidate_id=candidate_id,
        behavior_type=behavior_type,
        source="test",
        target_id=candidate_id.split(":", 1)[-1],
        target_name=candidate_id,
        goal_xyyaw=[x, y, 0.0],
        interaction_command=interaction,
        features={
            "exploration_gain": gain,
            "distance_m": abs(x),
            "interaction_cost": 1.0,
            "confidence": 0.9,
        },
        metadata={
            "target_room_id": room_id,
            "target_goal": target_goal,
            "node_type": node_type,
            "state": state,
            "expected_effect": (
                "access_room" if node_type == "portal" else "reveal_contents"
            ),
            "connected_room_ids": [1, 2] if node_type == "portal" else [],
            "frontier_point": [x, y],
            "cell_count": int(10 + gain * 10),
            "map_resolution": 0.05,
            "hard_constraints_passed": True,
        },
    )


def test_curator_uses_type_quotas_instead_of_global_rule_score() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(
            candidate_top_k=8,
            navigate_quota=1,
            interaction_quota=3,
            explore_quota=4,
            max_frontiers_per_room=2,
        )
    )
    candidates = [
        make_candidate("target:apple", "NAVIGATE", x=5.0, target_goal=True),
        *[
            make_candidate(
                f"interaction:container_{index}:open",
                "INTERACT",
                x=1.0 + index,
                node_type="container",
            )
            for index in range(5)
        ],
        *[
            make_candidate(
                f"frontier:{index}",
                "EXPLORE",
                x=10.0 + index,
                y=float(index),
                room_id=1 if index < 3 else 2,
                gain=1.0 - index * 0.05,
            )
            for index in range(6)
        ],
    ]

    result = curator.curate(candidates)

    counts = {
        behavior_type: sum(
            candidate.behavior_type == behavior_type
            for candidate in result.candidates
        )
        for behavior_type in ("NAVIGATE", "INTERACT", "EXPLORE")
    }
    assert counts == {"NAVIGATE": 1, "INTERACT": 3, "EXPLORE": 4}
    assert "target:apple" in result.mandatory_ids


def test_curator_hard_rejects_invalid_or_already_satisfied_candidates() -> None:
    curator = CandidateCurator()
    unsafe = make_candidate("frontier:unsafe", "EXPLORE", x=1.0)
    unsafe.metadata["hard_constraints_passed"] = False
    already_open = make_candidate(
        "interaction:fridge:open",
        "INTERACT",
        x=2.0,
        node_type="container",
        state="open",
    )

    accepted, rejected = curator.filter_candidates([unsafe, already_open])

    assert accepted == []
    assert rejected == {
        "frontier:unsafe": "hard_constraints_failed",
        "interaction:fridge:open": "interaction_action_already_satisfied",
    }


def test_spatial_history_survives_frontier_reclustering_and_suppresses_low_gain() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(candidate_top_k=4, repeat_guard_low_gain_limit=2)
    )
    old = make_candidate("frontier:old", "EXPLORE", x=1.10, y=1.10)
    reclustered = make_candidate("frontier:new", "EXPLORE", x=1.20, y=1.15)
    alternative = make_candidate("frontier:other", "EXPLORE", x=3.10, y=1.10)
    old_key = candidate_history_key(old, region_size_m=1.0)

    assert candidate_history_key(reclustered, region_size_m=1.0) == old_key
    result = curator.curate(
        [reclustered, alternative],
        history_by_key={old_key: {"selection_count": 3, "low_gain_repeat_count": 2}},
    )

    assert [candidate.candidate_id for candidate in result.candidates] == [
        "frontier:other"
    ]
    assert result.omitted["frontier:new"] == "history_low_gain_suppressed"


def test_curator_suppresses_low_visible_area_while_large_opening_exists() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(
            candidate_top_k=4,
            explore_quota=4,
            explore_min_visible_gain_ratio=0.25,
        )
    )
    narrow = make_candidate("frontier:narrow", "EXPLORE", x=1.0, gain=0.9)
    wide = make_candidate("frontier:wide", "EXPLORE", x=4.0, gain=1.0)
    narrow.metadata["expected_visible_unknown_area_m2"] = 2.0
    wide.metadata["expected_visible_unknown_area_m2"] = 20.0

    result = curator.curate([narrow, wide])

    assert [candidate.candidate_id for candidate in result.candidates] == [
        "frontier:wide"
    ]
    assert (
        result.omitted["frontier:narrow"]
        == "low_expected_visible_gain_suppressed"
    )


def test_candidate_update_accepts_benign_sequence_refresh() -> None:
    selected = make_candidate("frontier:a", "EXPLORE", x=1.0, y=2.0)
    latest = make_candidate("frontier:a", "EXPLORE", x=1.1, y=2.0)

    validation = validate_candidate_update(selected, [latest])

    assert validation.valid
    assert validation.candidate is latest


def test_candidate_update_rejects_moved_or_changed_interaction() -> None:
    selected = make_candidate(
        "interaction:fridge:open",
        "INTERACT",
        x=1.0,
        node_type="container",
    )
    moved = make_candidate(
        "interaction:fridge:open",
        "INTERACT",
        x=2.0,
        node_type="container",
    )
    changed = make_candidate(
        "interaction:fridge:open",
        "INTERACT",
        x=1.0,
        node_type="container",
    )
    changed.interaction_command["interaction_group_id"] = "left_door"

    assert validate_candidate_update(selected, [moved]).reason == "candidate_goal_moved"
    assert (
        validate_candidate_update(selected, [changed]).reason
        == "candidate_semantics_changed"
    )
