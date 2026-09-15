import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("distance,accepted", [(1.69, False), (1.95, False), (1.5, False), (1.2, True)])
@pytest.mark.parametrize("historical_only", [False, True])
def test_graph_verification_checks_live_object_distance(distance, accepted, historical_only):
    source = Path(__file__).resolve().parents[1] / "scripts/semantic_behavior_executor.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_verify_graph_locked")
    scope = {"math": math, "STATE_VERIFYING": "VERIFYING"}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    machine = SimpleNamespace(
        state="VERIFYING",
        on_verification_result=lambda success, detail: [(success, detail)],
        on_target_visibility=lambda visible, detail: [(visible, detail)],
    )
    executor = SimpleNamespace(
        machine=machine, map_frame="map", _current_pose=lambda frame: (0., 0., 0.),
        selection={"target_id": "target", "behavior_type": "NAVIGATE", "metadata": {
            "target_goal": True, "target_success_distance_threshold_m": 1.5,
            "target_object_distance_m": .5, "target_navigation_required": False,
            "target_require_current_visibility": not historical_only,
        }},
        latest_graph={"nodes": [{"id": "target", "aabb_center": [distance, 0., .5],
            "is_currently_visible": not historical_only,
            "attributes": {"visible_pixels": 0 if historical_only else 100,
            "visible_fraction": 0 if historical_only else .8,
            "consecutive_observations": 0 if historical_only else 5,
            "max_visible_pixels": 100, "max_visible_fraction": .8,
            "max_consecutive_observations": 5}}]},
    )
    result = scope["_verify_graph_locked"](executor)
    assert result[0][0] is accepted
    if not accepted:
        assert result[0][1]["reason"] == "target_distance_not_satisfied"
        assert result[0][1]["target_object_distance_m"] == distance
