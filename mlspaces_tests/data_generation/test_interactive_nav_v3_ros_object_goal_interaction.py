"""Unit coverage for evaluator-owned opaque ROS object interactions.

This deliberately uses only fakes: the public method command contains an
opaque instance ID, while raw simulator names/joints remain available only in
the evaluator-private score record.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.InteractiveNav.evaluation import benchmark_metrics, benchmark_runner
from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import EvaluatorInteractionRequest
from scripts.InteractiveNav.evaluation.trusted_interaction_skill import (
    JointOpenResult,
    ObjectInteractionResult,
    OpaqueObjectRegistry,
    OpenPostcondition,
    OpenPostconditionSpec,
    TrustedInteractionSkill,
)


class _FakeAdapter:
    """Minimal evaluator-side adapter fake; it never imports or starts ROS."""

    def __init__(self, request: EvaluatorInteractionRequest) -> None:
        self._request: EvaluatorInteractionRequest | None = request
        self.completions: list[tuple[str, bool]] = []
        self.outcomes: list[dict] = []

    def pop_next_interaction_request(self) -> EvaluatorInteractionRequest | None:
        request, self._request = self._request, None
        return request

    def complete_interaction(
        self,
        command_id: str,
        *,
        success: bool,
        status: str | None = None,
        reason: str | None = None,
        outcome: dict | None = None,
    ) -> dict[str, str]:
        self.completions.append((command_id, success))
        self.outcomes.append(dict(outcome or {}))
        result = {"status": status or ("COMPLETED" if success else "FAILED")}
        if reason:
            result["reason"] = reason
        if outcome:
            result.update(
                {
                    key: value
                    for key, value in outcome.items()
                    if key
                    in {
                        "state",
                        "pre_state",
                        "post_state",
                        "interaction_capability",
                        "interactable",
                        "retryable",
                        "failure_reason",
                        "verification_source",
                        "interaction_pose_validation",
                        "physics_substeps",
                        "task_steps_consumed",
                    }
                }
            )
        return result


def _runtime_joint(
    *,
    object_name: str,
    joint_name: str,
    joint_index: int,
    body_id: int = 1,
    domain: str = "container",
) -> benchmark_runner.RuntimeJoint:
    return benchmark_runner.RuntimeJoint(
        object_name=object_name,
        object_category="Fridge",
        domain=domain,
        joint_name=joint_name,
        joint_index=joint_index,
        body_id=body_id,
        aabb_center=np.asarray([0.0, 0.0, 0.0]),
        aabb_size=np.asarray([1.0, 1.0, 1.0]),
    )


def _public_pose_env(*, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    pose = np.eye(4, dtype=float)
    pose[0, 0] = np.cos(yaw)
    pose[0, 1] = -np.sin(yaw)
    pose[1, 0] = np.sin(yaw)
    pose[1, 1] = np.cos(yaw)
    pose[0, 3] = x
    pose[1, 3] = y
    return SimpleNamespace(
        current_robot=SimpleNamespace(
            robot_view=SimpleNamespace(base=SimpleNamespace(pose=pose))
        )
    )


def _valid_public_interaction_fields(
    *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0
) -> dict:
    """Return method/public-frame evidence for one local interaction."""

    return {
        "public_command": {
            "interaction_approach_pose_xyyaw": [x, y, yaw],
            "interaction_ready_distance_m": 0.45,
            "interaction_ready_yaw_tolerance_rad": 0.55,
        },
        "public_observation": {
            "capture_step": 7,
            "age_seconds": 0.1,
            "box_3d": {
                "center": [x, y, 1.0],
                "size": [1.0, 1.0, 2.0],
                "frame_id": "world",
            },
        },
    }


def test_restricted_gt_root_body_alias_resolves_to_articulated_object_skill() -> None:
    """A rendered door frame must route to its child hinge's opaque skill."""

    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0, 1, 2, 0]),
        body_rootid=np.asarray([0, 1, 1, 1, 4]),
    )
    leaf = _runtime_joint(
        object_name="private_door_leaf",
        joint_name="private_hinge",
        joint_index=0,
        body_id=2,
    )
    aliases = benchmark_runner._perception_source_skill_aliases(
        model=model,
        private_specs=[
            SimpleNamespace(source_name="private_door_root", body_id=1),
            SimpleNamespace(source_name="private_door_leaf", body_id=2),
            SimpleNamespace(source_name="private_door_handle", body_id=3),
            SimpleNamespace(source_name="unrelated_object", body_id=4),
        ],
        joints_by_object={"private_door_leaf": [leaf]},
    )

    assert aliases == {
        "private_door_root": "private_door_leaf",
        "private_door_leaf": "private_door_leaf",
        "private_door_handle": "private_door_leaf",
    }


def test_door_instance_stem_fallback_resolves_when_body_roots_disagree() -> None:
    """Door root/leaf asset suffixes can differ while the private skill is unique."""

    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0, 0]),
        body_rootid=np.asarray([0, 1, 2]),
    )
    leaf = _runtime_joint(
        object_name="doorway_hash_1_2_2",
        joint_name="private_hinge",
        joint_index=0,
        body_id=2,
        domain="channel",
    )
    aliases = benchmark_runner._perception_source_skill_aliases(
        model=model,
        private_specs=[SimpleNamespace(source_name="doorway_hash_1_0_2", body_id=1)],
        joints_by_object={leaf.object_name: [leaf]},
    )
    assert aliases == {"doorway_hash_1_0_2": "doorway_hash_1_2_2"}


def test_door_instance_stem_fallback_keeps_same_asset_instances_separate() -> None:
    """The fallback must not merge two doorway copies sharing one asset hash."""

    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0, 0, 0, 0]),
        body_rootid=np.asarray([0, 1, 2, 3, 4]),
    )
    first_leaf = _runtime_joint(
        object_name="doorway_hash_1_2_2",
        joint_name="first_hinge",
        joint_index=0,
        body_id=2,
        domain="channel",
    )
    second_leaf = _runtime_joint(
        object_name="doorway_hash_2_2_2",
        joint_name="second_hinge",
        joint_index=1,
        body_id=4,
        domain="channel",
    )
    aliases = benchmark_runner._perception_source_skill_aliases(
        model=model,
        private_specs=[
            SimpleNamespace(source_name="doorway_hash_1_0_2", body_id=1),
            SimpleNamespace(source_name="doorway_hash_2_0_2", body_id=3),
        ],
        joints_by_object={
            first_leaf.object_name: [first_leaf],
            second_leaf.object_name: [second_leaf],
        },
    )
    assert aliases == {
        "doorway_hash_1_0_2": "doorway_hash_1_2_2",
        "doorway_hash_2_0_2": "doorway_hash_2_2_2",
    }


def test_restricted_gt_door_root_opaque_id_is_registered_for_the_leaf_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public frame's root-body ID must be accepted by the sealed adapter."""

    startup_events: list[tuple[str, str]] = []

    class _FakeAdapter:
        def __init__(self, **_kwargs) -> None:
            self.private_instances: dict[str, str] = {}
            self.instance_aliases: dict[str, str] = {}

        def reset(
            self,
            *,
            episode_id,
            private_instances,
            instance_aliases=None,
            **_kwargs,
        ) -> None:
            startup_events.append(("adapter_reset", str(episode_id)))
            self.private_instances = dict(private_instances)
            self.instance_aliases = dict(instance_aliases or {})

        def publish_restricted_gt_frame(self, *_args, **_kwargs) -> None:
            return None

    class _FakeGoalStatusObserver:
        def begin_episode(self, episode_id: str) -> None:
            startup_events.append(("goal_begin", str(episode_id)))

    # Deliberately place the render/root body and the articulated leaf in
    # separate MuJoCo roots; this mirrors the ProcTHOR doorway asset seam.
    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0, 0, 0]),
        body_rootid=np.asarray([0, 1, 2, 3]),
    )
    leaf = _runtime_joint(
        object_name="doorway_hash_1_2_2",
        joint_name="private_hinge",
        joint_index=0,
        body_id=2,
        domain="channel",
    )
    fridge = _runtime_joint(
        object_name="private_fridge",
        joint_name="private_fridge_hinge",
        joint_index=1,
        body_id=3,
    )
    freezer = _runtime_joint(
        object_name="private_fridge",
        joint_name="private_freezer_hinge",
        joint_index=2,
        body_id=3,
    )
    specs = [
        SimpleNamespace(source_name="doorway_hash_1_0_2", body_id=1),
        SimpleNamespace(source_name="doorway_hash_1_2_2", body_id=2),
        SimpleNamespace(source_name="private_fridge", body_id=3),
    ]
    monkeypatch.setattr(benchmark_runner, "RosObjectGoalEvaluatorAdapter", _FakeAdapter)
    monkeypatch.setattr(
        benchmark_runner,
        "build_private_object_specs_from_env",
        lambda _env: specs,
    )
    monkeypatch.setattr(
        benchmark_runner.RestrictedGTPerceptionPublisher,
        "build",
        lambda self, *_args, **_kwargs: None,
    )

    task = SimpleNamespace(env=SimpleNamespace(current_model=model))
    catalog = SimpleNamespace(joints=[leaf, fridge, freezer])
    config = SimpleNamespace(
        restricted_gt_min_visible_pixels=16,
        restricted_gt_min_bbox_area_pixels=512,
        restricted_gt_max_distance_m=4.0,
        ros_target_topic="/target",
        ros_restricted_gt_topic="/gt",
        ros_interaction_command_topic="/command",
        ros_interaction_result_topic="/result",
    )
    runtime = benchmark_runner._build_restricted_ros_object_goal_runtime(
        task=task,
        catalog=catalog,
        episode={
            "interactive_nav": {
                "target": {"selected_instance": "private_target"},
                "interactions": [
                    {"object_name": "private_fridge", "joint_index": 1}
                ],
                "oracle_plans": [
                    {"required_interaction_ids": ["private_recipe_fridge"]}
                ],
            }
        },
        public=SimpleNamespace(instruction="find the apple"),
        config=config,
        episode_index=0,
        goal_status_observer=_FakeGoalStatusObserver(),
    )
    alternate_runtime = benchmark_runner._build_restricted_ros_object_goal_runtime(
        task=task,
        catalog=catalog,
        episode={
            "interactive_nav": {
                "target": {"selected_instance": "private_target"},
                "interactions": [
                    {"object_name": "private_fridge", "joint_index": 2}
                ],
                "oracle_plans": [
                    {"required_interaction_ids": ["private_recipe_freezer"]}
                ],
            }
        },
        public=SimpleNamespace(instruction="find the apple"),
        config=config,
        episode_index=0,
    )

    root_opaque_id = runtime.perception.registry.public_id_for("doorway_hash_1_0_2")
    leaf_opaque_id = runtime.perception.registry.public_id_for("doorway_hash_1_2_2")
    assert runtime.opaque_to_source_name[root_opaque_id] == "doorway_hash_1_2_2"
    assert runtime.opaque_to_joints[root_opaque_id] == (leaf,)
    assert runtime.opaque_to_source_name[leaf_opaque_id] == "doorway_hash_1_2_2"
    assert root_opaque_id in runtime.adapter.private_instances
    assert leaf_opaque_id in runtime.adapter.private_instances
    assert runtime.adapter.instance_aliases[
        benchmark_runner.opaque_door_instance_id(root_opaque_id)
    ] == root_opaque_id
    assert startup_events[0] == ("goal_begin", runtime.perception.episode_id)
    assert startup_events[1] == ("adapter_reset", runtime.perception.episode_id)
    assert runtime.adapter.instance_aliases[
        benchmark_runner.opaque_door_instance_id(leaf_opaque_id)
    ] == leaf_opaque_id
    fridge_opaque_id = runtime.perception.registry.public_id_for("private_fridge")
    alternate_fridge_id = alternate_runtime.perception.registry.public_id_for(
        "private_fridge"
    )
    assert runtime.opaque_to_joints[fridge_opaque_id] == (fridge, freezer)
    assert alternate_runtime.opaque_to_joints[alternate_fridge_id] == (
        fridge,
        freezer,
    )
    assert benchmark_runner.opaque_door_instance_id(fridge_opaque_id) not in runtime.adapter.instance_aliases


def test_opaque_ros_object_command_keeps_private_resolution_out_of_public_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sealed multi-joint skill credits V3 rows without leaking their IDs."""

    opaque_id = "obj_000017"
    command_id = "request_opaque_001"
    raw_object_name = "private_fridge_body_42"
    raw_joint_names = ("private_fridge_hinge_left", "private_fridge_hinge_right")
    scoring_ids = ("v3_internal_open_left", "v3_internal_open_right")
    joints = (
        _runtime_joint(object_name=raw_object_name, joint_name=raw_joint_names[0], joint_index=11),
        _runtime_joint(object_name=raw_object_name, joint_name=raw_joint_names[1], joint_index=12),
    )
    registry = OpaqueObjectRegistry()
    registry.register(
        opaque_id,
        joints=joints,
        object_ref=raw_object_name,
        open_postcondition=OpenPostconditionSpec(success_fraction=0.8, minimum_open_joints=1),
    )
    executed_joint_names: list[str] = []

    def execute_open_joint(joint: benchmark_runner.RuntimeJoint) -> JointOpenResult:
        # These names are deliberately private to the trusted executor.
        executed_joint_names.append(joint.joint_name)
        result_by_index = {
            11: JointOpenResult(True, 0.0, 0.90, simulated_seconds=0.35),
            12: JointOpenResult(True, 0.0, 0.85, simulated_seconds=0.65),
        }
        return result_by_index[joint.joint_index]

    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_1",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            **_valid_public_interaction_fields(),
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=TrustedInteractionSkill(registry, execute_open_joint),
        opaque_to_source_name={opaque_id: raw_object_name},
        opaque_to_joints={opaque_id: joints},
    )
    task = SimpleNamespace(
        env=_public_pose_env(),
        get_observations=lambda: {"camera": "public-observation"},
    )
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
        force_max_internal_steps=1500,
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": scoring_ids[0],
                    "object_name": raw_object_name,
                    "joint_index": 11,
                    "prerequisites": [],
                },
                {
                    "interaction_id": scoring_ids[1],
                    "object_name": raw_object_name,
                    "joint_index": 12,
                    "prerequisites": [],
                },
            ],
            "oracle_plans": [
                {
                    "plan_id": "private_full_open_plan",
                    "required_interaction_ids": list(scoring_ids),
                }
            ],
        }
    }
    published_steps: list[int] = []
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: pytest.fail(
            "restricted execution must not consult private access geometry"
        ),
    )
    monkeypatch.setattr(benchmark_runner, "_robot_lock_snapshot", lambda _env: object())
    monkeypatch.setattr(benchmark_runner, "_apply_robot_lock", lambda *_args: None)

    def execute_group(_env, *, object_name, joints, **_kwargs):
        assert object_name == raw_object_name
        executed_joint_names.extend(joint.joint_name for joint in joints)
        return SimpleNamespace(
            success=True,
            joint_results=(
                JointOpenResult(True, 0.0, 0.90, simulated_seconds=1.0),
                JointOpenResult(True, 0.0, 0.85, simulated_seconds=0.0),
            ),
            simulated_seconds=1.0,
            physics_substeps=500,
            pre_state="closed",
            post_state="open",
            private_metadata={"execution_mode": "ordinary_force_group_fast"},
        )

    monkeypatch.setattr(
        benchmark_runner,
        "execute_open_articulation_group",
        execute_group,
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda _runtime, _task, *, decision_index: published_steps.append(decision_index) or True,
    )
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=8,
        frames=[],
    )

    assert consumed is not None
    private_attempt = consumed["private_attempt"]
    public_attempt = consumed["public_attempt"]
    assert executed_joint_names == list(raw_joint_names)
    assert adapter.completions == [(command_id, True)]
    assert adapter.outcomes[-1]["interaction_capability"] == "articulated"
    assert adapter.outcomes[-1]["failure_reason"] == ""
    assert private_attempt["resolved_object_name"] == raw_object_name
    assert private_attempt["resolved_interaction_ids"] == list(scoring_ids)
    assert private_attempt["resolved_interaction_id"] == scoring_ids[0]
    assert private_attempt["simulated_seconds"] == pytest.approx(1.0)
    assert consumed["simulated_seconds"] == pytest.approx(1.0)
    assert published_steps == [8]

    # This is the policy-visible projection.  It carries only opaque routing,
    # high-level outcome and elapsed skill time; raw object/joint/V3 IDs remain
    # exclusively in ``private_attempt`` above.
    assert public_attempt["request_id"] == command_id
    assert public_attempt["instance_id"] == opaque_id
    assert public_attempt["operation"] == "open"
    assert public_attempt["status"] == "COMPLETED"
    assert public_attempt["state"] == "open"
    assert public_attempt["post_state"] == "open"
    assert public_attempt["interaction_capability"] == "articulated"
    assert public_attempt["decision_step"] == 8
    assert public_attempt["simulated_seconds"] == pytest.approx(1.0)
    assert public_attempt["result_status"] == "COMPLETED"
    public_json = json.dumps(public_attempt, sort_keys=True)
    for private_value in (raw_object_name, *raw_joint_names, *scoring_ids):
        assert private_value not in public_json
    assert {"resolved_object_name", "resolved_joint_name", "resolved_interaction_id"}.isdisjoint(public_attempt)

    # The plural private IDs are what allow one sealed object action to satisfy
    # its V3 multi-joint plan.  Patch final joint fractions only; no simulator is
    # created for this regression test.
    monkeypatch.setattr(benchmark_metrics, "joint_open_fraction", lambda _env, _row: 1.0)
    terminal = benchmark_metrics.score_interactions(object(), episode, [private_attempt])
    assert terminal.valid_plan_id == "private_full_open_plan"
    assert terminal.required_interaction_success is True
    assert terminal.correct_action_count == 2


def test_unknown_semantic_portal_is_scored_as_invalid_not_failed_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_id = "open-static-doorway"
    opaque_id = "obj_000021"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_invalid_portal",
            instance_id=opaque_id,
            action="open",
            private_handle=None,
            node_id="portal_obj_000021",
            candidate_id="interaction:portal_obj_000021:open",
            rejection_reason="unknown_instance_id",
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=object(),
        opaque_to_source_name={},
        opaque_to_joints={},
    )
    task = SimpleNamespace(
        env=_public_pose_env(),
        get_observations=lambda: {"camera": "public-observation"},
    )
    config = SimpleNamespace(record_video=False)
    published_steps: list[int] = []
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda _runtime, _task, *, decision_index: published_steps.append(decision_index) or True,
    )
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode={"interactive_nav": {"interactions": [], "oracle_plans": []}},
        private_attempts=[],
        config=config,
        decision_index=12,
        frames=[],
    )

    assert consumed is not None
    assert adapter.completions == [(command_id, False)]
    assert published_steps == [12]
    assert consumed["private_attempt"]["classification"] == "invalid"
    assert consumed["private_attempt"]["success"] is False
    assert consumed["private_attempt"]["metadata"]["rejection"] == {
        "reason": "unknown_instance_id",
        "node_id": "portal_obj_000021",
        "candidate_id": "interaction:portal_obj_000021:open",
        "rejected_before_execution": True,
        "action_skipped": False,
        "public_state": "unavailable",
    }
    public_attempt = consumed["public_attempt"]
    assert public_attempt["request_id"] == command_id
    assert public_attempt["instance_id"] == opaque_id
    assert public_attempt["operation"] == "open"
    assert public_attempt["status"] == "INVALID"
    assert public_attempt["state"] == "unavailable"
    assert public_attempt["interaction_capability"] == "unavailable"
    assert public_attempt["reason"] == "unknown_instance_id"
    assert public_attempt["decision_step"] == 12
    assert public_attempt["simulated_seconds"] == 0.0
    assert public_attempt["result_status"] == "INVALID"


def test_not_visible_rejection_is_neutral_and_does_not_reveal_private_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_id = "known-but-not-visible"
    opaque_id = "obj_000077"
    source_name = "private_fridge"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_visibility_gate",
            instance_id=opaque_id,
            action="open",
            private_handle=None,
            rejection_reason="interaction_not_visible",
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={},
    )
    task = SimpleNamespace(
        env=_public_pose_env(),
        get_observations=lambda: {"camera": "public-observation"},
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode={
            "interactive_nav": {
                "interactions": [
                    {
                        "interaction_id": "required_fridge_open",
                        "object_name": source_name,
                        "prerequisites": [],
                    }
                ],
                "oracle_plans": [],
            }
        },
        private_attempts=[],
        config=SimpleNamespace(record_video=False),
        decision_index=3,
        frames=[],
    )

    assert consumed is not None
    private_attempt = consumed["private_attempt"]
    public_attempt = consumed["public_attempt"]
    assert private_attempt["classification"] == "invalid"
    assert private_attempt["success"] is False
    assert private_attempt["resolved_object_name"] is None
    assert private_attempt["resolved_interaction_ids"] == []
    assert public_attempt["status"] == "FAILED"
    assert public_attempt["failure_reason"] == "interaction_not_visible"
    assert public_attempt["interaction_capability"] == "unknown"
    assert "interactable" not in public_attempt
    assert public_attempt["retryable"] is True


def test_drawer_scan_scoring_ids_follow_physical_order_without_capability_gate() -> None:
    source_name = "private_dresser_body"
    top = _runtime_joint(object_name=source_name, joint_name="top_slide", joint_index=9)
    bottom = _runtime_joint(object_name=source_name, joint_name="bottom_slide", joint_index=4)
    # Deliberately store the V3 rows in the opposite order.  The macro's
    # top-to-bottom physical scan order must be what prerequisite scoring sees.
    episode = {
        "interactive_nav": {
            "interactions": [
                {
                    "interaction_id": "drawer_bottom",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                },
                {
                    "interaction_id": "drawer_top",
                    "object_name": source_name,
                    "joint_index": 9,
                    "type": "container_sliding_drawer",
                },
            ]
        }
    }

    assert benchmark_runner._successful_drawer_scan_interaction_ids(
        episode=episode,
        source_name=source_name,
        opened_joints=(top, bottom),
    ) == ["drawer_top", "drawer_bottom"]

    # Frozen V3 types affect only private scoring labels; they never decide
    # whether the public drawer action is executable.
    episode["interactive_nav"]["interactions"][0]["type"] = "container_hinged_door"
    episode["interactive_nav"]["interactions"][1]["type"] = "container_hinged_door"
    assert benchmark_runner._successful_drawer_scan_interaction_ids(
        episode=episode,
        source_name=source_name,
        opened_joints=(top, bottom),
    ) == ["drawer_top", "drawer_bottom"]


def test_static_open_portal_trace_matches_public_completion_without_force_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StaticOpenAdapter(_FakeAdapter):
        def complete_interaction(self, command_id: str, **_kwargs):
            self.completions.append((command_id, False))
            self.outcomes.append({})
            return {
                "status": "SUCCEEDED",
                "success": True,
                "state": "static_open",
                "interaction_capability": "static",
                "interactable": False,
                "retryable": False,
            }

    command_id = "skip-static-aperture"
    request = EvaluatorInteractionRequest(
        command_id=command_id,
        episode_id="episode_static_open",
        instance_id="obj_000099",
        action="open",
        private_handle=None,
        node_id="portal_obj_000099",
        candidate_id="interaction:portal_obj_000099:open",
        rejection_reason="unknown_instance_id",
    )
    runtime = SimpleNamespace(
        adapter=_StaticOpenAdapter(request),
        opaque_to_source_name={},
        opaque_to_joints={},
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=SimpleNamespace(env=object(), get_observations=lambda: {}),
        runtime=runtime,
        episode={"interactive_nav": {"interactions": [], "oracle_plans": []}},
        private_attempts=[],
        config=SimpleNamespace(record_video=False),
        decision_index=1,
        frames=[],
    )

    assert consumed is not None
    assert consumed["public_attempt"]["status"] == "SUCCEEDED"
    assert consumed["public_attempt"]["success"] is True
    rejection = consumed["private_attempt"]["metadata"]["rejection"]
    assert rejection["action_skipped"] is True
    assert rejection["public_state"] == "static_open"
    # The evaluator did not apply force, so the private physical attempt stays
    # false even though the public aperture postcondition is already satisfied.
    assert consumed["private_attempt"]["success"] is False


def test_private_scoring_orders_effect_ids_by_oracle_plan() -> None:
    source_name = "private_fridge_body"
    episode = {
        "interactive_nav": {
            "interactions": [
                {
                    "interaction_id": "outer",
                    "object_name": source_name,
                    "joint_index": 3,
                },
                {
                    "interaction_id": "inner",
                    "object_name": source_name,
                    "joint_index": 1,
                    "prerequisites": [{"interaction_id": "outer"}],
                },
                {
                    "interaction_id": "alternative",
                    "object_name": source_name,
                    "joint_index": 0,
                },
            ],
            "oracle_plans": [
                {"plan_id": "two_joint", "required_interaction_ids": ["outer", "inner"]},
                {"plan_id": "alternative", "required_interaction_ids": ["alternative"]},
            ],
        }
    }

    assert benchmark_runner._order_interaction_ids_by_oracle_plan(
        episode,
        ["inner", "outer"],
    ) == ["outer", "inner"]


def test_partial_object_skill_result_is_reported_failed_to_ros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "private_fridge_body"
    opaque_id = "obj_000008"
    command_id = "partial_open"
    joints = (
        _runtime_joint(object_name=source_name, joint_name="outer", joint_index=3),
        _runtime_joint(object_name=source_name, joint_name="inner", joint_index=1),
    )
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_000008",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            **_valid_public_interaction_fields(),
        )
    )
    public_result = ObjectInteractionResult(
        request_id=command_id,
        instance_id=opaque_id,
        operation="open",
        status="completed",
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=SimpleNamespace(
            execute_private=lambda _request: SimpleNamespace(
                joint_results=(
                    JointOpenResult(True, 0.0, 0.9),
                    JointOpenResult(False, 0.0, 0.2),
                ),
                public_result=public_result,
                postcondition=OpenPostcondition.SATISFIED,
            )
        ),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: joints},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "outer",
                    "object_name": source_name,
                    "joint_index": 3,
                    "prerequisites": [],
                },
                {
                    "interaction_id": "inner",
                    "object_name": source_name,
                    "joint_index": 1,
                    "prerequisites": [{"interaction_id": "outer"}],
                },
            ],
            "oracle_plans": [
                {"plan_id": "both", "required_interaction_ids": ["outer", "inner"]}
            ],
        }
    }
    task = SimpleNamespace(env=_public_pose_env(), get_observations=lambda: {})
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
        force_max_internal_steps=1500,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: pytest.fail(
            "restricted execution must not consult private access geometry"
        ),
    )
    monkeypatch.setattr(benchmark_runner, "_robot_lock_snapshot", lambda _env: object())
    monkeypatch.setattr(benchmark_runner, "_apply_robot_lock", lambda *_args: None)
    monkeypatch.setattr(
        benchmark_runner,
        "execute_open_articulation_group",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            joint_results=(
                JointOpenResult(True, 0.0, 0.9),
                JointOpenResult(False, 0.0, 0.2),
            ),
            simulated_seconds=0.0,
            physics_substeps=50,
            pre_state="closed",
            post_state="blocked",
            private_metadata={"execution_mode": "ordinary_force_group_fast"},
        ),
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=1,
        frames=[],
    )

    assert consumed is not None
    assert adapter.completions == [(command_id, False)]
    assert adapter.outcomes[-1]["interaction_capability"] == "blocked"
    assert adapter.outcomes[-1]["failure_reason"] == "force_target_not_reached"
    assert adapter.outcomes[-1]["verification_source"] == "executor_state_verification"
    assert consumed["public_attempt"]["status"] == "FAILED"
    assert consumed["public_attempt"]["result_status"] == "FAILED"
    assert consumed["private_attempt"]["classification"] == "required_valid"
    assert consumed["private_attempt"]["success"] is False
    assert consumed["private_attempt"]["resolved_interaction_ids"] == ["outer"]


def test_unsafe_refrigerator_sweep_matches_ordinary_retry_contract() -> None:
    outcome = benchmark_runner._public_force_execution_outcome(
        execution=SimpleNamespace(
            success=False,
            physics_substeps=0,
            pre_state="closed",
        ),
        executor_metadata={
            "public_failure_reason": "unsafe_open_sweep",
            "task_steps_consumed": 0,
            "open_sweep_preflight": {
                "checked": True,
                "safe": False,
                "recommended_retreat_m": 0.25,
            },
        },
        pose_validation={"checked": True, "valid": True},
    )

    assert outcome == {
        "state": "unknown",
        "pre_state": "closed",
        "post_state": "unknown",
        "interaction_capability": "articulated",
        "interactable": True,
        "retryable": True,
        "failure_reason": "unsafe_open_sweep",
        "verification_source": "executor_open_sweep_preflight",
        "interaction_pose_validation": {"checked": True, "valid": True},
        "physics_substeps": 0,
        "task_steps_consumed": 0,
        "execution_cost": 0.0,
        "recommended_retreat_m": 0.25,
    }


def test_failed_drawer_scan_cannot_return_transient_target_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque_id = "obj_000071"
    source_name = "private_dresser_body"
    joint = _runtime_joint(object_name=source_name, joint_name="private_slide", joint_index=4)
    command_id = "drawer_scan_failure"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_2",
            instance_id=opaque_id,
            action="scan",
            private_handle=object(),
            sequence_type="drawer_scan",
            open_regions=((0.5, 0.2),),
            **_valid_public_interaction_fields(),
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=object(),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "drawer_target",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                    "prerequisites": [],
                }
            ],
            "oracle_plans": [],
        }
    }
    task = SimpleNamespace(
        env=_public_pose_env(),
        get_observations=lambda: {"camera": "public-observation"},
    )
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: pytest.fail(
            "restricted execution must not consult private access geometry"
        ),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_execute_private_drawer_scan",
        lambda **_kwargs: {
            "success": False,
            "joint_results": (),
            "opened_joints": (),
            "simulated_seconds": 1.0,
            # Simulate a buggy future executor.  The V3 boundary must still
            # reject its transient evidence when the macro has failed.
            "target_discovery": {
                "distance_m": 0.4,
                "visibility_fraction": 0.1,
                "visible_pixels": 32,
                "group_index": 0,
            },
            "metadata": {},
            "observation": {"camera": "public-observation"},
        },
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=4,
        frames=[],
    )

    assert consumed is not None
    assert adapter.completions == [(command_id, False)]
    assert consumed["private_attempt"]["success"] is False
    assert consumed["target_discovery"] is None


def test_drawer_open_lifecycle_uses_grounded_groups_and_restores_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_name = "private_dresser"
    joints = (
        _runtime_joint(
            object_name=source_name,
            joint_name="private_slide_a",
            joint_index=2,
        ),
        _runtime_joint(
            object_name=source_name,
            joint_name="private_slide_b",
            joint_index=3,
        ),
    )
    grouping_calls: list[dict] = []

    def fake_groups(task, available, regions, **kwargs):
        grouping_calls.append(dict(kwargs))
        assert tuple(available) == joints
        assert regions == ((0.25, 0.25), (0.75, 0.75))
        return [
            (joints[0], {"grounding_source": "m1_region"}),
            (joints[1], {"grounding_source": "m1_region"}),
        ]

    monkeypatch.setattr(benchmark_runner, "_drawer_scan_runtime_groups", fake_groups)
    monkeypatch.setattr(
        benchmark_runner.probe,
        "get_head_joint_position",
        lambda _env: np.asarray([0.1, 0.2]),
    )
    monkeypatch.setattr(
        benchmark_runner.probe,
        "get_torso_joint_position",
        lambda _env: np.asarray([0.3, 0.4]),
    )
    monkeypatch.setattr(
        benchmark_runner.probe,
        "lower_head_for_drawer_view",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        benchmark_runner.probe,
        "lean_torso_for_drawer_view",
        lambda *_args, **_kwargs: None,
    )
    restored: list[str] = []
    monkeypatch.setattr(
        benchmark_runner.probe,
        "set_head_joint_position",
        lambda *_args, **_kwargs: restored.append("head"),
    )
    monkeypatch.setattr(
        benchmark_runner.probe,
        "set_torso_joint_position",
        lambda *_args, **_kwargs: restored.append("torso"),
    )
    monkeypatch.setattr(benchmark_runner.mujoco, "mj_forward", lambda *_args: None)
    monkeypatch.setattr(
        benchmark_runner,
        "joint_open_fraction",
        lambda env, row: 0.95,
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    published: list[int] = []
    monkeypatch.setattr(
        benchmark_runner,
        "_publish_restricted_ros_frame",
        lambda runtime, task, *, decision_index: published.append(decision_index)
        or True,
    )
    executed: list[str] = []

    def execute_group(selected):
        joint = selected[0]
        executed.append(joint.joint_name)
        return (
            SimpleNamespace(
                success=True,
                joint_results=(JointOpenResult(True, 0.0, 0.95, 0.1),),
                simulated_seconds=0.1,
                physics_substeps=5,
                pre_state="closed",
            ),
            {"execution_mode": "ordinary_force_group_fast"},
        )

    env = _public_pose_env()
    env.current_model = object()
    env.current_data = object()
    env.camera_manager = SimpleNamespace(
        registry=SimpleNamespace(update_all_cameras=lambda _env: None)
    )
    task = SimpleNamespace(
        env=env,
        get_observations=lambda: {"head_camera": "fresh"},
    )

    result = benchmark_runner._execute_private_drawer_open(
        task=task,
        runtime=object(),
        joints=joints,
        open_regions=((0.25, 0.25), (0.75, 0.75)),
        allowed_joint_indices={2, 3},
        execute_group=execute_group,
        config=SimpleNamespace(record_video=False),
        decision_index=17,
        frames=[],
    )

    assert result["success"] is True
    assert executed == ["private_slide_a", "private_slide_b"]
    assert grouping_calls == [
        {"allowed_joint_indices": {2, 3}, "fallback_to_all": False}
    ]
    assert published == [17, 17, 17]
    assert restored == ["head", "torso"]
    assert result["metadata"]["view_profile"] == "drawer_low_view"
    assert result["metadata"]["final_state_satisfied"] is True


def test_drawer_open_grounding_failure_returns_public_skill_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_grounding(*_args, **_kwargs):
        raise RuntimeError("private drawer details must stay private")

    monkeypatch.setattr(
        benchmark_runner,
        "_drawer_scan_runtime_groups",
        fail_grounding,
    )
    task = SimpleNamespace(
        env=object(),
        get_observations=lambda: {"head_camera": "fallback"},
    )

    result = benchmark_runner._execute_private_drawer_open(
        task=task,
        runtime=object(),
        joints=(),
        open_regions=((0.5, 0.5),),
        allowed_joint_indices=set(),
        execute_group=lambda _joints: pytest.fail("force must not run"),
        config=SimpleNamespace(record_video=False),
        decision_index=18,
        frames=[],
    )

    assert result["success"] is False
    assert result["metadata"] == {
        "execution_mode": "ordinary_drawer_open_fast_sequence",
        "reason": "drawer_open_execution_failed",
        "error_type": "RuntimeError",
        "group_count": 0,
    }
    assert result["observation"] == {"head_camera": "fallback"}
    assert "private drawer details" not in json.dumps(result)


def test_wrong_public_approach_yaw_is_retryable_and_skips_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque_id = "obj_000091"
    source_name = "private_fridge"
    command_id = "wrong-yaw"
    joint = _runtime_joint(
        object_name=source_name,
        joint_name="private_hinge",
        joint_index=2,
    )
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_wrong_yaw",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            public_command={
                "interaction_approach_pose_xyyaw": [0.0, 0.0, 0.0],
                "interaction_ready_distance_m": 0.2,
                "interaction_ready_yaw_tolerance_rad": 0.2,
            },
            public_observation=_valid_public_interaction_fields()["public_observation"],
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    task = SimpleNamespace(
        env=_public_pose_env(yaw=1.0),
        get_observations=lambda: {},
    )
    monkeypatch.setattr(
        benchmark_runner,
        "execute_open_articulation_group",
        lambda *_args, **_kwargs: pytest.fail("force must not run at the wrong yaw"),
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode={"interactive_nav": {"interactions": [], "oracle_plans": []}},
        private_attempts=[],
        config=SimpleNamespace(record_video=False, interaction_max_distance_m=1.75),
        decision_index=2,
        frames=[],
    )

    assert consumed is not None
    outcome = adapter.outcomes[-1]
    assert outcome["failure_reason"] == "interaction_pose_invalid"
    assert outcome["verification_source"] == "executor_pose_precondition"
    assert outcome["retryable"] is True
    assert outcome["interaction_capability"] == "articulated"
    assert outcome["interaction_pose_validation"]["valid"] is False


def test_public_box_too_far_keeps_distinct_retryable_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opaque_id = "obj_000092"
    source_name = "private_fridge"
    joint = _runtime_joint(
        object_name=source_name,
        joint_name="private_hinge",
        joint_index=2,
    )
    public_fields = _valid_public_interaction_fields()
    public_fields["public_observation"]["box_3d"]["center"] = [5.0, 0.0, 1.0]
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id="too-far",
            episode_id="episode_too_far",
            instance_id=opaque_id,
            action="open",
            private_handle=object(),
            **public_fields,
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    task = SimpleNamespace(env=_public_pose_env(), get_observations=lambda: {})
    monkeypatch.setattr(
        benchmark_runner,
        "execute_open_articulation_group",
        lambda *_args, **_kwargs: pytest.fail("force must not run when public box is too far"),
    )
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode={"interactive_nav": {"interactions": [], "oracle_plans": []}},
        private_attempts=[],
        config=SimpleNamespace(record_video=False, interaction_max_distance_m=1.75),
        decision_index=2,
        frames=[],
    )

    assert consumed is not None
    outcome = adapter.outcomes[-1]
    assert outcome["failure_reason"] == "interaction_too_far"
    assert outcome["verification_source"] == "executor_pose_precondition"
    assert outcome["retryable"] is True


@pytest.mark.parametrize(
    ("direct_bbox_drawer_scan", "expected_fallback"),
    [(False, False), (True, False)],
)
def test_empty_drawer_regions_never_enumerate_hidden_slide_joints(
    monkeypatch: pytest.MonkeyPatch,
    direct_bbox_drawer_scan: bool,
    expected_fallback: bool,
) -> None:
    opaque_id = "obj_000071"
    source_name = "private_dresser_body"
    joint = _runtime_joint(object_name=source_name, joint_name="private_slide", joint_index=4)
    command_id = f"drawer_scan_empty_{int(direct_bbox_drawer_scan)}"
    adapter = _FakeAdapter(
        EvaluatorInteractionRequest(
            command_id=command_id,
            episode_id="episode_public_3",
            instance_id=opaque_id,
            action="scan",
            private_handle=object(),
            sequence_type="drawer_scan",
            open_regions=(),
            direct_bbox_drawer_scan=direct_bbox_drawer_scan,
            **_valid_public_interaction_fields(),
        )
    )
    runtime = SimpleNamespace(
        adapter=adapter,
        skill=object(),
        opaque_to_source_name={opaque_id: source_name},
        opaque_to_joints={opaque_id: (joint,)},
    )
    episode = {
        "interactive_nav": {
            "interaction_requirement": "required",
            "interactions": [
                {
                    "interaction_id": "drawer_target",
                    "object_name": source_name,
                    "joint_index": 4,
                    "type": "container_sliding_drawer",
                    "prerequisites": [],
                }
            ],
            "oracle_plans": [],
        }
    }
    task = SimpleNamespace(
        env=_public_pose_env(),
        get_observations=lambda: {"camera": "public-observation"},
    )
    config = SimpleNamespace(
        interaction_max_distance_m=1.75,
        require_interaction_visible=True,
        record_video=False,
    )
    scan_kwargs: dict = {}
    monkeypatch.setattr(
        benchmark_runner,
        "_check_interaction_access",
        lambda *_args, **_kwargs: pytest.fail(
            "restricted execution must not consult private access geometry"
        ),
    )

    def execute_scan(**kwargs):
        scan_kwargs.update(kwargs)
        success = bool(kwargs["fallback_to_all"])
        return {
            "success": success,
            "joint_results": (),
            "opened_joints": (joint,) if success else (),
            "simulated_seconds": 1.0 if success else 0.0,
            "target_discovery": None,
            "metadata": {"execution_mode": "drawer_scan"},
            "observation": {"camera": "public-observation"},
        }

    monkeypatch.setattr(benchmark_runner, "_execute_private_drawer_scan", execute_scan)
    monkeypatch.setattr(benchmark_runner, "_capture_head_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_runner, "_publish_restricted_ros_frame", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(benchmark_runner, "_discard_task_rollout_cache", lambda _task: None)

    consumed = benchmark_runner._consume_pending_ros_object_goal_interaction(
        task=task,
        runtime=runtime,
        episode=episode,
        private_attempts=[],
        config=config,
        decision_index=5,
        frames=[],
    )

    assert consumed is not None
    assert scan_kwargs["open_regions"] == ()
    assert scan_kwargs["fallback_to_all"] is expected_fallback
    assert consumed["private_attempt"]["success"] is expected_fallback
    assert adapter.completions == [(command_id, expected_fallback)]
