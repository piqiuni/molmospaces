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
        "scripts.InteractiveNav.force_interaction_bridge.open_door_root_with_force",
        lambda _env, root_name, config: {
            "root_body_name": root_name,
            "pre_state": "closed",
            "post_state": "open",
            "joint_infos": [
                {
                    "joint_name": "left_hinge",
                    "joint_type": "hinge",
                    "joint_range": [0.0, 1.0],
                    "joint_value": 0.99,
                    "open_fraction": 0.99,
                }
            ],
            "physics_substeps": 123,
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
    result = controller.before_step(SimpleNamespace(env=object()), step=17)

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
