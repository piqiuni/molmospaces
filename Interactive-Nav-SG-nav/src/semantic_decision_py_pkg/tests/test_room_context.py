from __future__ import annotations

import pytest

from semantic_decision_py_pkg.room_context import room_context_for_xy


def _room(room_id, *, center=(0.0, 0.0), size=(4.0, 4.0), **attributes):
    return {
        "id": f"room_{room_id}",
        "room_id": room_id,
        "type": "room",
        "aabb_center": list(center),
        "aabb_size": list(size),
        "attributes": attributes,
    }


def test_inactive_room_does_not_claim_robot_position():
    graph = {"nodes": [_room(1, size=(1.0, 1.0), active=False), _room(2)]}

    assert room_context_for_xy(graph, (0.0, 0.0)) == {
        "room_id": 2,
        "potential_room": False,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_observed_room_precedes_overlapping_smaller_potential_room(reverse):
    nodes = [_room(1, size=(1.0, 1.0), is_potential_room=True), _room(2)]
    if reverse:
        nodes.reverse()

    assert room_context_for_xy({"nodes": nodes}, (0.0, 0.0))["room_id"] == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_smallest_containing_observed_room_wins_independent_of_node_order(reverse):
    nodes = [_room(1), _room(2, size=(2.0, 2.0))]
    if reverse:
        nodes.reverse()

    assert room_context_for_xy({"nodes": nodes}, (0.0, 0.0))["room_id"] == 2


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_containing_rooms_use_stable_room_id_tiebreak(reverse):
    nodes = [_room(2), _room(1)]
    if reverse:
        nodes.reverse()

    assert room_context_for_xy({"nodes": nodes}, (0.0, 0.0))["room_id"] == 1


def test_potential_room_is_preserved_when_it_is_the_only_containing_room():
    graph = {
        "nodes": [
            _room(1, center=(10.0, 10.0)),
            _room(2, is_potential_room=True, source_portal_id="portal_3"),
        ]
    }

    assert room_context_for_xy(graph, (0.0, 0.0)) == {
        "room_id": 2,
        "potential_room": True,
        "source_portal_id": "portal_3",
    }


@pytest.mark.parametrize("xy", [(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)])
def test_aabb_boundary_is_inside_room(xy):
    assert room_context_for_xy({"nodes": [_room(1)]}, xy)["room_id"] == 1


def test_outside_all_rooms_does_not_fall_back_to_nearest_center():
    assert room_context_for_xy({"nodes": [_room(1)]}, (3.0, 0.0)) == {}


@pytest.mark.parametrize("missing_key", ["aabb_center", "aabb_size"])
def test_missing_room_geometry_returns_unknown(missing_key):
    room = _room(1)
    room.pop(missing_key)

    assert room_context_for_xy({"nodes": [room]}, (0.0, 0.0)) == {}


@pytest.mark.parametrize("xy", [None, [], [0.0]])
def test_missing_robot_position_returns_unknown(xy):
    assert room_context_for_xy({"nodes": [_room(1)]}, xy) == {}


def test_empty_graph_returns_unknown():
    assert room_context_for_xy({"nodes": []}, (0.0, 0.0)) == {}
