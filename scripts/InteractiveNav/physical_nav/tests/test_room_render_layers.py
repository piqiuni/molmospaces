"""Room cuts are background pixels; interaction outlines/captions stay above."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import offline_semantic_renderer as rendering


def test_cabinet_projection_only_merges_same_room_and_preserves_source():
    import copy
    nodes = [{"id": "room_1", "type": "room", "room_id": 1}]
    for ident, label, room in [("a", "cabinet", 1), ("b", "cabinet", 1),
                               ("c", "cabinet", 2), ("d", "fridge", 1), ("e", "locker", 1)]:
        nodes.append({"id": ident, "type": "container", "label": label, "room_id": room})
    graph = {"nodes": nodes, "edges": [
        {"src_id": "room_1", "dst_id": key, "relation": "has_child"} for key in ("a", "b", "d", "e")
    ] + [{"src_id": "b", "dst_id": "food", "relation": "contains"}]}
    original = copy.deepcopy(graph)
    projected = rendering.aggregate_room_cabinets(graph)
    assert graph == original
    aggregate = next(n for n in projected["nodes"] if n.get("display_count"))
    assert aggregate["display_count"] == 2
    assert set(aggregate["display_member_ids"]) == {"a", "b"}
    assert {n["id"] for n in projected["nodes"]} >= {"c", "d", "e"}
    assert sum(e["dst_id"] == aggregate["id"] for e in projected["edges"]) == 1
    assert any(e["src_id"] == aggregate["id"] and e["dst_id"] == "food" for e in projected["edges"])


def test_idle_header_explains_retry_and_blocked_frontiers(monkeypatch):
    trace = {"eligibility_rejections": {
        "interaction:door:open": "candidate_cooldown",
        "frontier:1": "frontier_region_failed_requires_topology_change"}}
    step = {"semantic_selection": {"active": False}, "semantic_decision_trace": trace}
    assert rendering.idle_subgoal_status(step) == ("WAIT:", "interaction retry cooldown")
    captured = []
    monkeypatch.setattr(rendering.SubgoalOverlay, "draw_header", lambda *a, **k: captured.append(a))
    rendering.draw_task_subgoal_header(np.zeros((200, 600, 3), np.uint8), step)
    assert captured[0][2:4] == ("WAIT:", "interaction retry cooldown")
    del trace["eligibility_rejections"]["interaction:door:open"]
    assert rendering.idle_subgoal_status(step)[0] == "STOP:"
    trace["execution_eligible_candidate_count"] = 1
    assert rendering.idle_subgoal_status(step) == ("WAIT:", "selecting subgoal")


def test_boxes_and_cabinets_aggregate_separately_per_room():
    nodes = [{"id": str(i), "type": "container", "label": label, "room_id": room}
             for i, (label, room) in enumerate([
                 ("box", 1), ("box", 1), ("cabinet", 1), ("cabinet", 1),
                 ("box", 2), ("box", None), ("fridge", 1), ("fridge", 1)])]
    result = rendering.aggregate_room_cabinets({"nodes": nodes, "edges": []})
    grouped = {n["label"]: n["display_count"] for n in result["nodes"] if n.get("display_count")}
    assert grouped == {"box": 2, "cabinet": 2}
    assert {n["id"] for n in result["nodes"] if not n.get("display_count")} == {"4", "5", "6", "7"}


def test_room_cuts_then_all_boxes_then_object_captions(monkeypatch):
    renderer = rendering.OfflineSixPanelRenderer(
        transforms=rendering.TransformResolver([], map_frame="map", odom_frame="odom")
    )
    occ = rendering.RawGrid(np.zeros((40, 40), np.int32), 40, 40, .1, "map", -2, -2, 0)
    rooms = np.full((40, 40), 1, np.int32)
    rooms[:, 20] = -1
    rooms[:, 21:] = 2
    room = rendering.RawGrid(rooms, 40, 40, .1, "map", -2, -2, 0)
    events = []
    original_blend = rendering.cv2.addWeighted
    original_lines = rendering.cv2.polylines
    original_text = rendering.cv2.putText

    def blend(*args, **kwargs):
        events.append("underlay")
        return original_blend(*args, **kwargs)

    def lines(*args, **kwargs):
        events.append("box")
        return original_lines(*args, **kwargs)

    def text(*args, **kwargs):
        if "cabinet" in args[1]:
            events.append("caption")
        return original_text(*args, **kwargs)

    monkeypatch.setattr(rendering.cv2, "addWeighted", blend)
    monkeypatch.setattr(rendering.cv2, "polylines", lines)
    monkeypatch.setattr(rendering.cv2, "putText", text)
    step = {"observed_instance_ids": ["container_1", "container_2"], "unified_graph": {"nodes": [
        {"id": "container_1", "type": "container", "label": "cabinet",
         "aabb_center": [0, 0, 1], "aabb_size": [1, .4, 2]},
        {"id": "container_2", "type": "container", "label": "cabinet",
         "aabb_center": [.2, .2, 1], "aabb_size": [1, .4, 2]},
    ]}}
    panel = renderer.render_room_panel(occ, room, (400, 400), step, 1, (-2, -2, 2, 2))
    assert panel.shape == (400, 400, 3)
    assert events == ["underlay", "underlay", "box", "box", "caption", "caption"]
    image = rendering._room_base(room)
    assert tuple(image[10, 10]) != tuple(image[10, 30])
