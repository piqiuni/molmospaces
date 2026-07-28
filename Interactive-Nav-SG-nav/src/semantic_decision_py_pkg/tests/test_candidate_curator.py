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


def test_curator_prioritizes_reachable_unvisited_room_frontier() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(candidate_top_k=4, explore_quota=4)
    )
    historical = make_candidate("frontier:history", "EXPLORE", x=8.0, room_id=2)
    visited = make_candidate("frontier:a_visited", "EXPLORE", x=3.0, room_id=2)
    unreachable = make_candidate(
        "frontier:b_unreachable", "EXPLORE", x=3.0, y=1.0, room_id=4
    )
    new_room = make_candidate(
        "frontier:z_new_room", "EXPLORE", x=3.0, y=-1.0, room_id=3
    )
    for candidate in (visited, unreachable, new_room):
        candidate.metadata.update(
            {
                "robot_room_id": 1,
                "room_reachable": True,
                "expected_visible_unknown_area_m2": 10.0,
            }
        )
    unreachable.metadata["room_reachable"] = False

    result = curator.curate(
        [visited, unreachable, new_room],
        history_by_key={
            candidate_history_key(historical): {"selection_count": 1},
        },
    )

    assert result.ranked_ids_by_type["EXPLORE"][0] == "frontier:z_new_room"
    assert (
        result.quality_terms_by_id["frontier:z_new_room"]
        ["unvisited_room_frontier_bonus"]
        == 0.15
    )
    assert (
        result.quality_terms_by_id["frontier:b_unreachable"]
        ["unvisited_room_frontier_bonus"]
        == 0.0
    )
    assert result.decision_hint_by_id["frontier:z_new_room"] == "NEW_ROOM_FRONTIER"
    assert "frontier:b_unreachable" not in result.decision_hint_by_id


def test_curator_rejects_invisible_unreachable_container_interaction() -> None:
    candidate = make_candidate(
        "interaction:container_stale:open",
        "INTERACT",
        x=4.0,
        node_type="container",
    )
    candidate.metadata.update(
        {
            "is_currently_visible": False,
            "room_reachable": False,
        }
    )

    result = CandidateCurator().curate([candidate])

    assert result.candidates == []
    assert result.rejected == {
        "interaction:container_stale:open": "container_not_visible_and_room_unreachable"
    }


def test_curator_keeps_substantially_larger_visible_area_ahead_of_new_room() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(candidate_top_k=3, explore_quota=3)
    )
    historical = make_candidate("frontier:history", "EXPLORE", x=8.0, room_id=2)
    large_visited = make_candidate(
        "frontier:z_large_visited", "EXPLORE", x=4.0, room_id=2
    )
    smaller_new = make_candidate(
        "frontier:a_smaller_new", "EXPLORE", x=4.0, y=1.0, room_id=3
    )
    for candidate, area in ((large_visited, 20.0), (smaller_new, 8.0)):
        candidate.metadata.update(
            {
                "robot_room_id": 1,
                "room_reachable": True,
                "expected_visible_unknown_area_m2": area,
            }
        )

    result = curator.curate(
        [large_visited, smaller_new],
        history_by_key={
            candidate_history_key(historical): {"selection_count": 1},
        },
    )

    assert result.ranked_ids_by_type["EXPLORE"] == [
        "frontier:z_large_visited",
        "frontier:a_smaller_new",
    ]
    assert (
        result.quality_terms_by_id["frontier:z_large_visited"]
        ["visible_unknown_area_priority"]
        == 0.5
    )
    assert (
        result.quality_terms_by_id["frontier:a_smaller_new"]
        ["unvisited_room_frontier_bonus"]
        == 0.15
    )


def test_curator_keeps_repeat_low_gain_suppression_with_new_room_bonus() -> None:
    curator = CandidateCurator(
        CandidateCuratorConfig(
            candidate_top_k=3,
            explore_quota=3,
            repeat_guard_low_gain_limit=2,
        )
    )
    repeated = make_candidate(
        "frontier:repeat_same_room", "EXPLORE", x=2.0, room_id=2
    )
    new_room = make_candidate(
        "frontier:new_room", "EXPLORE", x=4.0, room_id=3
    )
    for candidate, area in ((repeated, 20.0), (new_room, 10.0)):
        candidate.metadata.update(
            {
                "robot_room_id": 1,
                "room_reachable": True,
                "expected_visible_unknown_area_m2": area,
            }
        )
    repeated_key = candidate_history_key(repeated)

    result = curator.curate(
        [repeated, new_room],
        history_by_key={
            repeated_key: {
                "selection_count": 3,
                "low_gain_repeat_count": 2,
            },
        },
    )

    assert result.omitted["frontier:repeat_same_room"] == "history_low_gain_suppressed"
    assert [candidate.candidate_id for candidate in result.candidates] == [
        "frontier:new_room"
    ]
    assert (
        result.quality_terms_by_id["frontier:new_room"]
        ["unvisited_room_frontier_bonus"]
        == 0.15
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


def test_object_goal_curator_hides_semantically_conflicting_container() -> None:
    curator = CandidateCurator()
    dresser = make_candidate(
        "interaction:dresser:open",
        "INTERACT",
        x=1.0,
        node_type="container",
    )
    dresser.target_name = "obj_000003"
    dresser.metadata.update(
        {
            "semantic_name": "dresser",
            "target_match": False,
            "robot_room_id": 1,
        }
    )
    portal = make_candidate(
        "interaction:portal_12:open",
        "INTERACT",
        x=2.0,
        node_type="portal",
    )
    portal.target_id = "portal_12"
    portal.metadata.update({"robot_room_id": 1, "connected_room_ids": [1, 2]})

    result = curator.curate(
        [dresser, portal],
        target_context={"enabled": True, "target_name": "Irishpotato_123"},
    )

    assert [candidate.candidate_id for candidate in result.candidates] == [
        "interaction:portal_12:open"
    ]
    assert (
        result.omitted["interaction:dresser:open"]
        == "target_container_semantic_mismatch"
    )


def test_object_goal_curator_keeps_plausible_drawer_for_pencil() -> None:
    curator = CandidateCurator()
    dresser = make_candidate(
        "interaction:dresser:open",
        "INTERACT",
        x=1.0,
        node_type="container",
    )
    dresser.target_name = "ChestOfDrawers_asset"
    dresser.metadata.update(
        {
            "semantic_name": "dresser",
            "target_match": False,
            "robot_room_id": 1,
        }
    )
    frontier = make_candidate("frontier:other", "EXPLORE", x=3.0)

    result = curator.curate(
        [dresser, frontier],
        target_context={"enabled": True, "target_name": "pencil"},
    )

    assert "interaction:dresser:open" in {
        candidate.candidate_id for candidate in result.candidates
    }
    assert (
        result.decision_hint_by_id["interaction:dresser:open"]
        == "PLAUSIBLE_TARGET_CONTAINER"
    )


def test_topology_pre_score_prioritizes_first_portal_on_observed_target_route() -> None:
    curator = CandidateCurator()
    target = make_candidate(
        "target:apple",
        "NAVIGATE",
        x=8.0,
        room_id=3,
        target_goal=True,
    )
    target.metadata["robot_room_id"] = 1
    first_portal = make_candidate(
        "interaction:portal_12:open",
        "INTERACT",
        x=2.0,
        node_type="portal",
    )
    first_portal.target_id = "portal_12"
    first_portal.metadata.update(
        {"robot_room_id": 1, "target_room_id": 1, "connected_room_ids": [1, 2]}
    )
    second_portal = make_candidate(
        "interaction:portal_23:open",
        "INTERACT",
        x=5.0,
        node_type="portal",
    )
    second_portal.target_id = "portal_23"
    second_portal.metadata.update(
        {"robot_room_id": 1, "target_room_id": 2, "connected_room_ids": [2, 3]}
    )
    graph = {
        "nodes": [
            {
                "id": "portal_12",
                "type": "portal",
                "attributes": {"connected_room_ids": [1, 2]},
            },
            {
                "id": "portal_23",
                "type": "portal",
                "attributes": {"connected_room_ids": [2, 3]},
            },
        ]
    }

    result = curator.curate(
        [target, first_portal, second_portal],
        graph=graph,
        target_context={"enabled": True, "target_name": "apple"},
    )

    assert result.decision_hint_by_id[first_portal.candidate_id] == "NEXT_ROUTE_PORTAL"
    assert result.decision_hint_by_id[second_portal.candidate_id] == "REMOTE_PORTAL"
    assert (
        result.quality_by_id[first_portal.candidate_id]
        > result.quality_by_id[second_portal.candidate_id]
    )
    assert (
        result.quality_terms_by_id[first_portal.candidate_id]["topology_priority"]
        == 1.35
    )


def test_post_interaction_traversal_outranks_plausible_container() -> None:
    curator = CandidateCurator()
    traversal = make_candidate(
        "post_interaction_traversal:portal_12",
        "NAVIGATE",
        x=1.5,
        room_id=2,
    )
    traversal.metadata.update(
        {
            "post_interaction_traversal": True,
            "robot_room_id": 1,
            "portal_node_id": "portal_12",
        }
    )
    dresser = make_candidate(
        "interaction:dresser:open",
        "INTERACT",
        x=0.5,
        node_type="container",
    )
    dresser.target_name = "dresser"
    dresser.metadata.update(
        {
            "semantic_name": "dresser",
            "target_match": False,
            "robot_room_id": 1,
        }
    )

    result = curator.curate(
        [traversal, dresser],
        target_context={"enabled": True, "target_name": "pencil"},
    )

    assert (
        result.decision_hint_by_id[traversal.candidate_id]
        == "POST_INTERACTION_TRAVERSE"
    )
    assert (
        result.quality_by_id[traversal.candidate_id]
        > result.quality_by_id[dresser.candidate_id]
    )
    assert (
        result.quality_terms_by_id[traversal.candidate_id]["topology_priority"]
        == 1.5
    )
