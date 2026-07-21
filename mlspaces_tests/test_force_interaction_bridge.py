from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "InteractiveNav"
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.InteractiveNav.force_interaction_bridge import AtomicForceInteractionController


def test_controller_emits_success_result_and_behavior_feedback(monkeypatch) -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    published = []
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.prepare_articulation_force",
        lambda _env, root_name, **_kwargs: {
            "group": {"root_body_name": root_name},
            "targets": {"left_hinge": 1.0},
            "selected_joint_names": ["left_hinge"],
            "closed_joint_names": [],
            "pre_joint_infos": [],
        },
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.drive_joint_group_to_targets",
        lambda _model, _data, _targets, config: {"physics_substeps": 123},
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.force_interaction_bridge.complete_articulation_force",
        lambda _env, _plan, config: {
            "pre_state": "closed",
            "post_state": "open",
            "joint_infos": [],
            "selected_joint_names": ["left_hinge"],
            "closed_joint_names": [],
            "success": True,
            "physics_substeps": 123,
            "task_steps_consumed": 1,
        },
    )
    monkeypatch.setattr(controller, "_publish", lambda publisher, payload: published.append(payload))

    assert controller.enqueue_command(
        {
            "command_id": "command_1",
            "candidate_id": "portal_1",
            "decision_id": "decision_1",
            "source_object_name": "double_door_root",
            "action": "open",
        }
    )
    task = SimpleNamespace(env=SimpleNamespace(current_model=object(), current_data=object()))
    assert controller.before_step(task, step=17) is None
    result = controller.after_step(task, step=17)

    assert result is not None
    assert result["success"] is True
    assert result["status"] == "SUCCEEDED"
    assert result["sim_steps_consumed"] == 1
    assert result["physics_substeps"] == 123
    assert result["source_object_name"] == "double_door_root"
    assert len(published) == 2
    assert published[0]["source"] == "force_atomic_interaction"
    assert published[1]["behavior_type"] == "INTERACT"
    assert published[1]["status"] == "SUCCEEDED"


def test_controller_deduplicates_command_ids() -> None:
    controller = AtomicForceInteractionController(close_all_doors_on_prepare=False)
    command = {
        "command_id": "same_command",
        "source_object_name": "door_root",
        "action": "open",
    }
    assert controller.enqueue_command(command)
    assert not controller.enqueue_command(command)
