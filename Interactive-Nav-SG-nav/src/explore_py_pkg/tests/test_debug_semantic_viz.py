from explore_py_pkg.debug_semantic_viz import (
    candidate_overlays,
    interaction_state_color,
    latest_state_change,
    portal_positions_between_rooms,
    portal_room_node_ids,
    room_style_by_id,
    topology_edge_style,
    topology_edge_visible,
    topology_order_rooms,
)


def test_candidate_overlays_prefers_unified_semantic_candidates():
    overlays = candidate_overlays(
        {
            "candidates": [
                {
                    "candidate_id": "interaction:door:open",
                    "behavior_type": "INTERACT",
                    "target_id": "door",
                    "target_name": "door",
                    "goal_xyyaw": [1.0, 2.0, 0.5],
                }
            ]
        },
        {
            "proposals": [
                {
                    "proposal_id": "frontier_1",
                    "goal_xyyaw": [5.0, 6.0, 0.0],
                }
            ]
        },
    )

    assert overlays == [
        {
            "candidate_id": "interaction:door:open",
            "behavior_type": "INTERACT",
            "target_id": "door",
            "target_name": "door",
            "goal_xyyaw": [1.0, 2.0, 0.5],
            "frame_id": "",
            "color": (255, 140, 0),
            "source": "semantic_decision",
        }
    ]


def test_candidate_overlays_falls_back_to_explore_proposals():
    overlays = candidate_overlays(
        {},
        {
            "frame_id": "tf_frame_map",
            "proposals": [
                {
                    "proposal_id": "frontier_1",
                    "goal_xyyaw": [5.0, 6.0, 0.25],
                }
            ],
        },
    )

    assert overlays[0]["candidate_id"] == "frontier:frontier_1"
    assert overlays[0]["behavior_type"] == "EXPLORE"
    assert overlays[0]["frame_id"] == "tf_frame_map"
    assert overlays[0]["color"] == (230, 40, 40)


def test_topology_edge_styles_keep_semantic_relations_distinct():
    styles = {
        "portal": topology_edge_style("connects", "portal", "room"),
        "container": topology_edge_style("has_child", "room", "container"),
        "support": topology_edge_style("supports", "support", "object"),
        "object": topology_edge_style("has_child", "room", "object"),
    }

    assert len({style["color"] for style in styles.values()}) == len(styles)
    assert topology_edge_style("adjacent_via", "room", "portal") is None


def test_latest_state_change_and_color():
    event = latest_state_change(
        [
            {"event": "NEW_NODE", "node_id": "door"},
            {"event": "STATE_CHANGED", "node_id": "door", "after": "open"},
        ]
    )

    assert event["node_id"] == "door"
    assert interaction_state_color(event["after"]) == (55, 185, 70)
    assert interaction_state_color("closed") == (55, 70, 225)


def test_topology_projection_only_keeps_interaction_hierarchy():
    nodes = {
        "room": {"type": "room"},
        "door": {"type": "portal"},
        "fridge": {"type": "container"},
        "apple": {"type": "object"},
        "chair": {"type": "object"},
        "table": {"type": "support"},
    }

    assert topology_edge_visible(
        {"src_id": "door", "dst_id": "room", "relation": "connects"}, nodes
    )
    assert topology_edge_visible(
        {"src_id": "room", "dst_id": "fridge", "relation": "has_child"}, nodes
    )
    assert topology_edge_visible(
        {"src_id": "fridge", "dst_id": "apple", "relation": "contains"}, nodes
    )
    assert not topology_edge_visible(
        {"src_id": "room", "dst_id": "chair", "relation": "has_child"}, nodes
    )
    assert not topology_edge_visible(
        {"src_id": "table", "dst_id": "chair", "relation": "supports"}, nodes
    )


def test_portal_room_ids_combine_inferred_attributes_and_graph_edges():
    portal = {
        "id": "door",
        "type": "portal",
        "attributes": {"connected_room_ids": [2]},
    }
    edges = [
        {"src_id": "door", "dst_id": "room_1", "relation": "connects"},
        {"src_id": "room_2", "dst_id": "door", "relation": "adjacent_via"},
    ]

    assert portal_room_node_ids(portal, edges) == ["room_1", "room_2"]


def test_connected_portal_is_positioned_between_its_rooms():
    portals = [
        {
            "id": "door",
            "type": "portal",
            "attributes": {"connected_room_ids": [1, 2]},
        }
    ]

    assert portal_positions_between_rooms(
        portals,
        [],
        {"room_1": (100, 120), "room_2": (300, 120)},
    ) == {"door": (200, 120)}

    assert portal_positions_between_rooms(
        portals,
        [],
        {"room_1": (100, 120), "room_2": (300, 120)},
        vertical_offset_y=35,
    ) == {"door": (200, 155)}


def test_topology_room_order_places_two_connection_hub_in_middle():
    rooms = [
        {"id": "room_1", "type": "room", "aabb_center": [0.0, 0.0, 0.0]},
        {"id": "room_3", "type": "room", "aabb_center": [1.0, 0.0, 0.0]},
        {"id": "room_2", "type": "room", "aabb_center": [2.0, 0.0, 0.0]},
    ]
    portals = [
        {"id": "door_12", "attributes": {"connected_room_ids": [1, 2]}},
        {"id": "door_23", "attributes": {"connected_room_ids": [2, 3]}},
    ]

    ordered = topology_order_rooms(rooms, portals)

    assert [room["id"] for room in ordered] == ["room_1", "room_2", "room_3"]


def test_room_style_labels_include_unknown_and_confidence():
    labels = room_style_by_id(
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "room_id": 1,
                    "label": "bedroom",
                    "attributes": {
                        "room_attribute": "livingroom",
                        "room_attribute_confidence": 0.526,
                    },
                },
                {
                    "id": "room_2",
                    "type": "room",
                    "room_id": 2,
                    "label": "unknown",
                    "attributes": {},
                },
            ]
        }
    )

    assert labels[1]["label"] == "Room 1 | livingroom | 0.53"
    assert labels[2]["label"] == "Room 2 | unknown | 0.00"
