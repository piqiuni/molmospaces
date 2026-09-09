from __future__ import annotations

from types import SimpleNamespace

from scripts.InteractiveNav.evaluation import benchmark_interaction_executor as executor


def test_multi_joint_container_uses_one_allowlisted_group_force(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def prepare(env, object_name, *, open_joint_names):
        calls.append(("prepare", (object_name, tuple(open_joint_names))))
        return {
            "pre_joint_infos": [
                {"joint_name": name, "open_fraction": 0.0}
                for name in open_joint_names
            ]
        }

    def complete(env, plan, *, config, robot_lock_callback):
        calls.append(("complete", config.max_physics_substeps))
        if robot_lock_callback is not None:
            robot_lock_callback()
        names = [row["joint_name"] for row in plan["pre_joint_infos"]]
        return {
            "success": True,
            "physical_success": True,
            "atomic_fallback": False,
            "physics_substeps": 25,
            "task_steps_consumed": 1,
            "pre_state": "closed",
            "post_state": "open",
            "pre_joint_infos": list(plan["pre_joint_infos"]),
            "joint_infos": [
                {"joint_name": name, "open_fraction": 1.0}
                for name in names
            ],
        }

    monkeypatch.setattr(executor, "prepare_articulation_state_force", prepare)
    monkeypatch.setattr(executor, "complete_articulation_force", complete)
    lock_calls: list[bool] = []
    env = SimpleNamespace(
        current_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
    )
    joints = [
        SimpleNamespace(joint_name="private_joint_a"),
        SimpleNamespace(joint_name="private_joint_b"),
        SimpleNamespace(joint_name="private_joint_c"),
    ]

    result = executor.execute_open_articulation_group(
        env,
        object_name="private_container",
        joints=joints,
        max_physics_substeps=1500,
        success_fraction=0.8,
        robot_lock_callback=lambda: lock_calls.append(True),
    )

    assert calls == [
        (
            "prepare",
            (
                "private_container",
                ("private_joint_a", "private_joint_b", "private_joint_c"),
            ),
        ),
        ("complete", 1500),
    ]
    assert lock_calls == [True]
    assert result.success is True
    assert result.physics_substeps == 25
    assert result.simulated_seconds == 0.05
    assert [row.executor_succeeded for row in result.joint_results] == [
        True,
        True,
        True,
    ]
    # Object-level time is not multiplied by the number of selected joints.
    assert sum(row.simulated_seconds for row in result.joint_results) == 0.05


def test_group_postcondition_requires_every_selected_joint(monkeypatch) -> None:
    def prepare(env, object_name, *, open_joint_names):
        return {
            "pre_joint_infos": [
                {"joint_name": name, "open_fraction": 0.0}
                for name in open_joint_names
            ]
        }

    def complete(env, plan, *, config, robot_lock_callback):
        names = [row["joint_name"] for row in plan["pre_joint_infos"]]
        return {
            "success": True,
            "physical_success": True,
            "atomic_fallback": False,
            "physics_substeps": 10,
            "task_steps_consumed": 1,
            "pre_state": "closed",
            "post_state": "ajar",
            "pre_joint_infos": list(plan["pre_joint_infos"]),
            "joint_infos": [
                {"joint_name": names[0], "open_fraction": 1.0},
                {"joint_name": names[1], "open_fraction": 0.2},
            ],
        }

    monkeypatch.setattr(executor, "prepare_articulation_state_force", prepare)
    monkeypatch.setattr(executor, "complete_articulation_force", complete)
    env = SimpleNamespace(
        current_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
    )

    result = executor.execute_open_articulation_group(
        env,
        object_name="private_container",
        joints=[
            SimpleNamespace(joint_name="private_joint_a"),
            SimpleNamespace(joint_name="private_joint_b"),
        ],
        max_physics_substeps=100,
        success_fraction=0.8,
    )

    assert result.success is False
    assert result.post_state == "blocked"


def test_refrigerator_open_uses_ordinary_sweep_preflight_and_applies_no_force(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def prepare(env, object_name, *, open_joint_names):
        calls.append("prepare")
        return {
            "pre_state": "closed",
            "pre_joint_infos": [
                {"joint_name": name, "open_fraction": 0.0}
                for name in open_joint_names
            ],
        }

    monkeypatch.setattr(executor, "prepare_articulation_state_force", prepare)
    monkeypatch.setattr(
        executor,
        "_requires_refrigerator_open_sweep",
        lambda command: command.get("container_kind") == "refrigerator",
    )
    monkeypatch.setattr(
        executor,
        "_refrigerator_open_sweep_preflight",
        lambda env, plan: {
            "checked": True,
            "safe": False,
            "reason": "unsafe_open_sweep",
            "recommended_retreat_m": 0.25,
        },
    )
    monkeypatch.setattr(
        executor,
        "complete_articulation_force",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe preflight must not apply force")
        ),
    )

    result = executor.execute_open_articulation_group(
        SimpleNamespace(
            current_model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        ),
        object_name="private_refrigerator",
        joints=[SimpleNamespace(joint_name="private_hinge")],
        max_physics_substeps=100,
        success_fraction=0.8,
        public_command={
            "node_type": "container",
            "action": "open",
            "container_kind": "refrigerator",
        },
    )

    assert calls == ["prepare"]
    assert result.success is False
    assert result.simulated_seconds == 0.0
    assert result.physics_substeps == 0
    assert result.pre_state == "closed"
    assert result.post_state == "closed"
    assert result.joint_results[0].error == "unsafe_open_sweep"
    assert result.private_metadata["public_failure_reason"] == "unsafe_open_sweep"
    assert result.private_metadata["open_sweep_preflight"][
        "recommended_retreat_m"
    ] == 0.25
