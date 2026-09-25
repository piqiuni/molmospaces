"""Pure policy regression: no ROS, wall sleeps, sensors or motion publishers."""

import ast
from dataclasses import FrozenInstanceError, replace
import math
from pathlib import Path

import pytest

from semantic_decision_py_pkg import behavior_execution as legacy
from semantic_decision_py_pkg import post_open_maps as maps
from semantic_decision_py_pkg.exit_observation import ExitObservationConfig, ExitObservationSweep


def receipts():
    baseline = maps.PostInteractionCostmapBaseline(
        "door", "event", 10, update_receipt_count=20,
        raw_occupancy_receipt_count=30, planning_occupancy_receipt_count=40,
        interaction_result_stamp_sec=100.0,
    )
    raw = maps.PostInteractionRawMapBarrier(
        31, 31, 100.1, 40, global_costmap_receipt_count=10,
        global_costmap_update_receipt_count=20,
    )
    planning = maps.PostInteractionPlanningMapBarrier(
        raw, "header_stamp", 41, 41, 100.1, "source_header_stamp", 10, 10, 20, 20,
    )
    snapshot = maps.MapReceiptSnapshot(
        maps.MapReceipt(31, 31, 100.1), maps.MapReceipt(41, 41, 100.1),
        maps.MapReceipt(10, 10, 100.0), maps.MapReceipt(21, 21, 100.2),
    )
    return baseline, raw, planning, snapshot


def test_map_api_compatibility_keeps_exact_class_and_function_identity():
    for name in ("PostInteractionCostmapBaseline", "PostInteractionRawMapBarrier",
                 "PostInteractionPlanningMapBarrier", "post_interaction_costmap_baseline_keys",
                 "post_interaction_raw_occupancy_fresh_source"):
        assert getattr(legacy, name) is getattr(maps, name)


@pytest.mark.parametrize("direct", [False, True])
def test_map_policy_requires_result_baseline_and_admitted_raw(direct):
    baseline, raw, planning, snapshot = receipts()
    policy = maps.PostOpenMapPolicy(direct)
    for args in ((None, raw, planning, snapshot), (baseline, None, planning, snapshot)):
        result = policy.evaluate(*args)
        assert not result.fresh and result.stage == "waiting_raw_occupancy"
        assert result.timeout_reason == "post_open_raw_occ_refresh_timeout"


def test_map_policy_matches_actual_planner_input_wiring():
    baseline, raw, planning, snapshot = receipts()
    strict = maps.PostOpenMapPolicy()
    direct = maps.PostOpenMapPolicy(direct_raw_costmap=True)
    assert strict.requires_planning_occupancy and not direct.requires_planning_occupancy
    result = strict.evaluate(baseline, raw, None, snapshot)
    assert result.stage == "waiting_planning_occupancy"
    assert result.timeout_reason == "post_open_planning_occ_refresh_timeout"
    result = strict.evaluate(baseline, raw, planning, snapshot)
    assert result.fresh and result.source == "costmap_update"
    result = direct.evaluate(baseline, raw, None, snapshot)
    assert result.fresh and result.source == "raw_occupancy_to_global_costmap_update"


@pytest.mark.parametrize("direct", [False, True])
def test_header_seq_change_alone_cannot_release_costmap_gate(direct):
    baseline, raw, planning, snapshot = receipts()
    snapshot = replace(snapshot, update=maps.MapReceipt(20, 999, 999.0))
    result = maps.PostOpenMapPolicy(direct).evaluate(baseline, raw, planning, snapshot)
    assert not result.fresh and result.stage == "waiting_global_costmap"
    assert result.timeout_reason == "post_open_costmap_refresh_timeout"
    with pytest.raises(FrozenInstanceError):
        snapshot.update.count = 1000


def test_direct_map_policy_rejects_older_source_even_with_a_new_receipt():
    baseline, raw, _, snapshot = receipts()
    old = replace(snapshot, update=maps.MapReceipt(21, 999, 99.0))
    assert not maps.PostOpenMapPolicy(True).evaluate(baseline, raw, None, old).fresh
    # A valid full publication remains an independent fallback.
    valid_full = replace(old, full=maps.MapReceipt(11, 1, 100.1))
    result = maps.PostOpenMapPolicy(True).evaluate(baseline, raw, None, valid_full)
    assert result.source == "raw_occupancy_to_global_costmap_full"


@pytest.mark.parametrize("direct", [False, True])
def test_map_diagnostics_keep_public_trace_schema(direct):
    baseline, raw, planning, snapshot = receipts()
    if direct:
        planning = None
    freshness = maps.PostOpenMapPolicy(direct).evaluate(baseline, raw, planning, snapshot)
    detail = maps.post_open_map_detail(
        {"target_id": "door"}, "event:event", baseline, raw, "header_stamp",
        planning, snapshot, freshness, elapsed_s=.25, timeout_s=5.0,
    )
    assert detail["opened_portal_id"] == "door"
    assert detail["post_open_costmap_baseline_key"] == "event:event"
    assert detail["post_open_raw_occ_fresh_source"] == "header_stamp"
    assert detail["post_open_result_stamp_sec"] == 100.0
    assert detail["post_open_causal_map_stage"] == "ready"
    assert detail["post_open_costmap_fresh"] and detail["costmap_fresh"]
    assert detail["baseline_global_costmap_seq"] == 20
    assert detail["observed_global_costmap_seq"] == 21
    assert detail["costmap_wait_elapsed"] == .25
    assert bool(detail.get("post_open_costmap_fast_path")) == direct


def test_strict_trace_uses_planning_admission_counter_not_result_counter():
    baseline, raw, planning, snapshot = receipts()
    planning = replace(planning, costmap_update_receipt_count=21)
    freshness = maps.PostOpenMapPolicy().evaluate(baseline, raw, planning, snapshot)
    detail = maps.post_open_map_detail(
        {}, "event:event", baseline, raw, "header_stamp", planning, snapshot,
        freshness, elapsed_s=1.0, timeout_s=5.0,
    )
    assert not freshness.fresh
    assert detail["baseline_global_costmap_seq"] == 21
    assert detail["post_open_causal_costmap_baseline_update_receipt_count"] == 21


@pytest.mark.parametrize("return_home", [False, True])
def test_sweep_visits_left_right_and_optional_home_without_sleep(return_home):
    config = ExitObservationConfig(return_home=return_home, settle_s=.1)
    sweep = ExitObservationSweep(0.0, config, now=0.0)
    step = sweep.advance(now=0.0, yaw=0.0)
    assert step.angular_z == -.3 and step.wait_s == .05
    assert sweep.targets[:2] == pytest.approx([-math.pi / 2, math.pi / 2])
    now = 0.0
    for target in sweep.targets:
        step = sweep.advance(now=now, yaw=target)
        assert step.phase == "settling" and step.angular_z == 0.0
        step = sweep.advance(now=now + .01, yaw=target)
        assert step.phase == "settling" and step.wait_s <= .05
        now += .11
        step = sweep.advance(now=now, yaw=target)
    assert step.result["status"] == "completed"
    assert len(step.result["views"]) == (3 if return_home else 2)
    assert step.angular_z == 0.0 and step.wait_s == 0.0
    # A consumer cannot mutate the sweep's sealed evidence through its report.
    step.result["views"].clear()
    assert len(sweep.advance(now=100, yaw=None).result["views"]) == len(sweep.targets)


def test_sweep_angle_wrap_speed_bound_and_missing_pose_stop():
    sweep = ExitObservationSweep(math.pi - .05, ExitObservationConfig(), now=1.0)
    step = sweep.advance(now=1.1, yaw=-math.pi + .05)
    assert step.angular_z == -.3
    assert all(-math.pi <= value <= math.pi for value in sweep.targets)
    for invalid in (None, float("nan")):
        step = sweep.advance(now=1.2, yaw=invalid)
        assert step.angular_z == 0.0 and step.result is None


def test_sweep_timeout_cancellation_and_shutdown_are_terminal():
    for kwargs, status in (({}, "timeout"), ({"current": False}, "canceled"),
                           ({"shutdown": True}, "canceled")):
        sweep = ExitObservationSweep(0.0, ExitObservationConfig(view_timeout_s=1.0), now=10.0)
        step = sweep.advance(now=11.0, yaw=0.0, **kwargs)
        assert step.result["status"] == status and step.angular_z == 0.0
        assert sweep.advance(now=12.0, yaw=-math.pi / 2).result == step.result


def test_sweep_cancels_during_settling_not_after_next_view():
    sweep = ExitObservationSweep(0.0, ExitObservationConfig(settle_s=10.0), now=0.0)
    assert sweep.advance(now=.1, yaw=-math.pi / 2).phase == "settling"
    step = sweep.advance(now=.15, yaw=-math.pi / 2, current=False)
    assert step.result["status"] == "canceled" and len(step.result["views"]) == 1


@pytest.mark.parametrize("fields", [{"speed_rad_s": 0}, {"view_timeout_s": -1},
                                   {"angle_rad": float("nan")}, {"settle_s": -1}])
def test_sweep_rejects_invalid_control_configuration(fields):
    with pytest.raises(ValueError):
        ExitObservationConfig(**fields)


def test_policy_modules_have_no_ros_threads_or_runtime_dependency():
    directory = Path(maps.__file__).parent
    for name in ("exit_observation.py", "post_open_maps.py"):
        tree = ast.parse((directory / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name == "math" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module in {"__future__", "dataclasses"}
