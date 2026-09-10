"""Standalone contracts for the post-evaluation top-down report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.InteractiveNav.evaluation import episode_topdown


class _FakeSceneMap:
    def __init__(self) -> None:
        self.occupancy = np.ones((8, 8), dtype=bool)
        self.room_map = None
        self.room_ids_to_name: dict[int, str] = {}
        self.px_per_m = 10

    def pos_px_to_m(self, row_col: np.ndarray) -> np.ndarray:
        values = np.asarray(row_col, dtype=float).reshape((-1, 2))
        return np.column_stack((values[:, 1], values[:, 0], np.zeros(len(values))))


def _episode() -> dict[str, Any]:
    return {
        "house_index": 7,
        "scene_dataset": "procthor-10k",
        "data_split": "val",
        "robot": {"init_qpos": {"base": [1.0, 1.0, 0.25]}},
        "task": {"pickup_obj_name": "alarm_clock_1"},
        "interactive_nav": {
            "case_id": "standalone-topdown",
            "interactions": [],
            "oracle_plans": [
                {
                    "steps": [
                        {
                            "type": "navigate",
                            "reason": "satisfy_nav_to_obj_success",
                            "goal_point": [5.0, 4.0],
                        }
                    ]
                }
            ],
        },
    }


def test_generated_scene_map_uses_tracked_core_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import molmo_spaces.utils.scene_maps as scene_maps

    model_path = tmp_path / "val_7.xml"
    generated = _FakeSceneMap()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        episode_topdown,
        "_resolve_scene_model_path",
        lambda *_args, **_kwargs: model_path,
    )
    monkeypatch.setattr(
        episode_topdown,
        "_resolve_precomputed_scene_map_path",
        lambda **_kwargs: None,
    )

    def fake_from_mj_model_path(**kwargs: Any) -> _FakeSceneMap:
        calls.append(kwargs)
        return generated

    monkeypatch.setattr(
        scene_maps.ProcTHORMap,
        "from_mj_model_path",
        staticmethod(fake_from_mj_model_path),
    )

    scene_map, resolved_model, radius, px_per_m, source, precomputed = (
        episode_topdown._load_static_scene_map(
            episode=_episode(),
            context=None,
            coverage_metadata={},
            scene_model_path=None,
        )
    )

    assert scene_map is generated
    assert resolved_model == model_path
    assert radius == pytest.approx(0.3)
    assert px_per_m == 200
    assert source == "generated_core_scene_map"
    assert precomputed is None
    assert calls == [
        {
            "model_path": str(model_path),
            "agent_radius": 0.3,
            "px_per_m": 200,
            "device_id": None,
        }
    ]


def test_render_uses_frozen_oracle_stages_without_method_scripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    benchmark_path = tmp_path / "benchmark.json"
    result_path = tmp_path / "episode_result.json"
    debug_dir = tmp_path / "debug"
    output_path = tmp_path / "episode_topdown.png"
    debug_dir.mkdir()
    benchmark_path.write_text(json.dumps([_episode()]), encoding="utf-8")
    result_path.write_text(
        json.dumps(
            {
                "result": {
                    "episode_index": 0,
                    "case_id": "standalone-topdown",
                    "terminal_reason": "policy_stop",
                    "navigation_path_length_m": 1.5,
                    "interaction_attempts": [],
                },
                "trace": [
                    {"base": {"base_pose_xyyaw": [1.0, 1.0, 0.25]}},
                    {"base": {"base_pose_xyyaw": [2.0, 2.0, 0.25]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        episode_topdown,
        "_load_static_scene_map",
        lambda **_kwargs: (
            _FakeSceneMap(),
            tmp_path / "val_7.xml",
            0.15,
            10,
            "test_complete_scene_map",
            None,
        ),
    )

    metadata = episode_topdown.render_episode_topdown(
        episode_result_path=result_path,
        benchmark_path=benchmark_path,
        debug_dir=debug_dir,
        output_path=output_path,
    )

    assert output_path.is_file() and output_path.stat().st_size > 0
    assert output_path.with_suffix(".json").is_file()
    assert metadata["schema_version"] == "interactive_nav_v3_episode_topdown_v4"
    assert metadata["scene_background"]["map_source"] == "test_complete_scene_map"
    assert metadata["gt_oracle_path"]["source"] == "frozen_oracle_plan_stages"
    assert metadata["gt_oracle_path"]["xy"] == [[1.0, 1.0], [5.0, 4.0]]
    assert metadata["gt_oracle_path_error"] is None
