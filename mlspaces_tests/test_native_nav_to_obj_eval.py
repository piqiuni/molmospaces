"""Focused tests for native nav-to-object screening controls."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for import_path in (REPO_ROOT, SCRIPT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.InteractiveNav.run_native_nav_to_obj_eval import (  # noqa: E402
    NativeRosBridgePolicy,
    _target_metadata,
    _resolve_distance_adaptive_horizon_steps,
)


@pytest.mark.parametrize(
    ("distance_m", "expected_steps"),
    [
        (0.0, 360),
        (10.0, 690),
        (100.0, 1000),
        (float("nan"), 1000),
    ],
)
def test_distance_adaptive_horizon_is_clamped_and_fail_safe(
    distance_m: float,
    expected_steps: int,
) -> None:
    assert (
        _resolve_distance_adaptive_horizon_steps(
            1000,
            distance_m,
            minimum_steps=360,
            fixed_overhead_steps=240,
            steps_per_meter=45.0,
        )
        == expected_steps
    )


class _FakeTask:
    def __init__(self, distance_m: float) -> None:
        self._task_horizon = 1000
        self.distance_m = distance_m

    def calculate_distance(self, index: int) -> float:
        assert index == 0
        return self.distance_m


def test_dynamic_horizon_preserves_original_cap_across_policy_resets() -> None:
    policy = object.__new__(NativeRosBridgePolicy)
    policy.task = _FakeTask(1.0)
    policy.native_dynamic_horizon_enabled = True
    policy.native_dynamic_horizon_min_steps = 360
    policy.native_dynamic_horizon_base_steps = 240
    policy.native_dynamic_horizon_steps_per_meter = 45.0
    policy.native_horizon_metadata = {}

    policy._configure_episode_horizon()

    assert policy.task._task_horizon == 360
    assert policy.task._native_base_task_horizon_steps == 1000

    policy.task.distance_m = 10.0
    policy._configure_episode_horizon()

    assert policy.task._task_horizon == 690
    assert policy.native_horizon_metadata["base_task_horizon_steps"] == 1000
    assert policy.native_horizon_metadata["effective_task_horizon_steps"] == 690


def test_target_metadata_uses_the_episode_success_distance_threshold() -> None:
    object_manager = SimpleNamespace(
        category_from_name=lambda _name: "laptop",
        fallback_expression=lambda _name: "laptop",
    )
    task = SimpleNamespace(
        config=SimpleNamespace(
            task_config=SimpleNamespace(
                pickup_obj_name="laptop_1",
                pickup_obj_candidates=["laptop_1"],
                selection_mode="specific_instance",
                succ_pos_threshold=1.25,
            )
        ),
        env=SimpleNamespace(object_managers=[object_manager], current_batch_index=0),
    )

    metadata = _target_metadata(task)

    assert metadata["target_context"]["success_distance_threshold_m"] == 1.25
