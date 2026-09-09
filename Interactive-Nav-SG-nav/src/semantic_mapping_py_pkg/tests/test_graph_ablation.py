from semantic_mapping_py_pkg.graph_ablation import apply_module1_ablation


def test_static_semantic_graph_removes_dynamic_object_state() -> None:
    graph = {
        "source_mode": "realtime_gt_observation",
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "confidence": 0.9,
                "is_currently_visible": True,
                "state_age_sec": 2.0,
                "attributes": {
                    "visible_pixels": 100,
                    "observation_evidence": {"joint_value": 0.0},
                },
                "interaction": {
                    "is_interactable": True,
                    "state": "closed",
                    "requires_interaction": True,
                    "operation_history": [{"success": True}],
                },
            }
        ],
    }
    result = apply_module1_ablation(graph, "static_semantic")
    node = result["nodes"][0]
    assert result["source_mode"] == "static_semantic_ablation"
    assert node["interaction"]["state"] == "unknown"
    assert node["interaction"]["operation_history"] == []
    assert node["is_currently_visible"] is False
    assert "observation_evidence" not in node["attributes"]
