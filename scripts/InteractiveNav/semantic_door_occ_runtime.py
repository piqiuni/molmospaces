from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.env.data_views import Door
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb


log = logging.getLogger(__name__)


def parse_pose_sequence(value: str) -> list[dict[str, Any]]:
    """Parse step:x,y,yaw,state,label entries separated by '/'."""
    phases = []
    for phase_index, raw_entry in enumerate(str(value or "").split("/")):
        entry = raw_entry.strip()
        if not entry:
            continue
        step_text, separator, payload = entry.partition(":")
        if not separator:
            raise ValueError(f"Invalid door OCC pose phase {entry!r}; expected step:x,y,yaw,state,label")
        values = [item.strip() for item in payload.split(",")]
        if len(values) != 5:
            raise ValueError(f"Invalid door OCC pose phase {entry!r}; expected five comma-separated fields")
        state = values[3].lower()
        if state not in {"closed", "open"}:
            raise ValueError(f"Invalid door state {state!r} in phase {entry!r}")
        phases.append(
            {
                "phase_index": phase_index,
                "step": int(step_text),
                "robot_xyyaw": [float(values[0]), float(values[1]), float(values[2])],
                "state": state,
                "label": values[4] or f"phase_{phase_index:02d}",
            }
        )
    if not phases:
        raise ValueError("Door OCC pose sequence is empty")
    if phases[0]["step"] != 0:
        raise ValueError("Door OCC pose sequence must start at step 0")
    if any(second["step"] <= first["step"] for first, second in zip(phases, phases[1:])):
        raise ValueError("Door OCC pose sequence steps must be strictly increasing")
    return phases


def _closed_value(joint_range) -> float:
    lower, upper = float(joint_range[0]), float(joint_range[1])
    if lower <= 0.0 <= upper:
        return 0.0
    return lower if abs(lower) <= abs(upper) else upper


def _open_value(joint_range, closed: float) -> float:
    lower, upper = float(joint_range[0]), float(joint_range[1])
    return lower if abs(lower - closed) >= abs(upper - closed) else upper


class DoorOccRuntimeController:
    """Hold all doors closed, then directly open one doorway root at a fixed step."""

    def __init__(self, root_name: str, open_step: int, output_path: str | Path):
        self.requested_root_name = str(root_name or "")
        self.open_step = max(0, int(open_step))
        self.output_path = Path(output_path).expanduser().resolve()
        self.target_root_id: int | None = None
        self.target_root_name = ""
        self.groups: dict[int, list[tuple[str, Door, int]]] = {}
        self.current_state = ""
        self._interaction_result_pub = None
        self.payload: dict[str, Any] = {
            "requested_root_name": self.requested_root_name,
            "open_step": self.open_step,
            "transitions": [],
        }

    def prepare(self, task) -> None:
        self._ensure_interaction_result_publisher()
        env = task.env
        model = env.current_model
        data = env.current_data
        object_manager = env.object_managers[env.current_batch_index]
        groups: dict[int, list[tuple[str, Door, int]]] = {}
        for door_name in object_manager.find_door_names():
            try:
                door = Door(door_name, data)
                hinge_index = door.get_hinge_joint_index()
            except (KeyError, ValueError):
                continue
            body_id = int(model.body(door_name).id)
            root_id = int(model.body_rootid[body_id])
            groups.setdefault(root_id, []).append((str(door_name), door, int(hinge_index)))
        if not groups:
            raise RuntimeError("Door OCC runtime test found no controllable doorway roots")

        root_names = {int(root_id): str(model.body(root_id).name or "") for root_id in groups}
        target_root_id = None
        if self.requested_root_name:
            for root_id, root_name in root_names.items():
                if root_name == self.requested_root_name:
                    target_root_id = root_id
                    break
            if target_root_id is None:
                raise RuntimeError(
                    f"Requested door root {self.requested_root_name!r} was not found; "
                    f"available roots={sorted(root_names.values())}"
                )
        else:
            robot_xy = env.current_robot.robot_view.base.pose[:2, 3]
            target_root_id = min(
                groups,
                key=lambda root_id: float(
                    (data.xpos[root_id][0] - robot_xy[0]) ** 2
                    + (data.xpos[root_id][1] - robot_xy[1]) ** 2
                ),
            )

        self.groups = groups
        self.target_root_id = int(target_root_id)
        self.target_root_name = root_names[self.target_root_id]

        initial_states = []
        for root_id in sorted(self.groups):
            initial_states.append(self._set_root_state(env, root_id, "closed"))
        mujoco.mj_forward(model, data)
        center, size = body_aabb(model, data, self.target_root_id, visual_only=True)
        base_pose = env.current_robot.robot_view.base.pose
        base_yaw = math.atan2(float(base_pose[1, 0]), float(base_pose[0, 0]))
        self.payload.update(
            {
                "target_root_name": self.target_root_name,
                "target_root_id": self.target_root_id,
                "target_closed_aabb_center": [float(value) for value in center],
                "target_closed_aabb_size": [float(value) for value in size],
                "robot_xyyaw": [
                    float(base_pose[0, 3]),
                    float(base_pose[1, 3]),
                    float(base_yaw),
                ],
                "initial_closed_roots": initial_states,
            }
        )
        self.current_state = "closed"
        self._record_transition(step=-1, state="closed", detail=initial_states[self._target_group_index()])
        self._publish_interaction_result(
            initial_states[self._target_group_index()],
            event_id="door_occ_initial_closed",
            action="close",
        )
        log.info(
            "Door OCC runtime test prepared: target=%s open_step=%d robot_xyyaw=%s",
            self.target_root_name,
            self.open_step,
            self.payload["robot_xyyaw"],
        )

    def before_step(self, task, step: int) -> bool:
        if self.target_root_id is None:
            return False
        desired_state = "open" if int(step) >= self.open_step else "closed"
        detail = self._set_root_state(task.env, self.target_root_id, desired_state)
        transitioned = desired_state != self.current_state
        corrected = bool(detail.get("joint_values_changed"))
        if transitioned:
            self.current_state = desired_state
            self._record_transition(step=int(step), state=desired_state, detail=detail)
            self._publish_interaction_result(
                detail,
                event_id=f"door_occ_step_{int(step):06d}_{desired_state}",
                action=desired_state,
            )
            log.info(
                "Door OCC runtime transition: step=%d root=%s state=%s",
                int(step),
                self.target_root_name,
                desired_state,
            )
        return transitioned or corrected

    def finalize(self, completed_steps: int) -> None:
        self.payload["completed_steps"] = int(completed_steps)
        self._write_payload()

    def _target_group_index(self) -> int:
        root_ids = sorted(self.groups)
        return root_ids.index(self.target_root_id)

    def _set_root_state(self, env, root_id: int, state: str) -> dict[str, Any]:
        transitions = []
        any_changed = False
        for door_name, door, hinge_index in self.groups[int(root_id)]:
            joint_range = door.get_joint_range(hinge_index)
            closed = _closed_value(joint_range)
            target = _open_value(joint_range, closed) if state == "open" else closed
            before = float(door.get_joint_position(hinge_index))
            changed = abs(before - target) > 1e-6
            if changed:
                door.set_joint_position(hinge_index, target)
                any_changed = True
            transitions.append(
                {
                    "leaf_body_name": door_name,
                    "joint_name": str(door.joint_names[hinge_index]),
                    "joint_range": [float(value) for value in joint_range],
                    "before": before,
                    "target": float(target),
                    "changed": changed,
                }
            )
        if any_changed:
            mujoco.mj_forward(env.current_model, env.current_data)
        return {
            "root_body_name": str(env.current_model.body(int(root_id)).name or f"root_{root_id}"),
            "state": state,
            "joint_values_changed": any_changed,
            "leaf_transitions": transitions,
        }

    def _record_transition(self, step: int, state: str, detail: dict[str, Any]) -> None:
        self.payload["transitions"].append(
            {
                "step": int(step),
                "state": str(state),
                "wall_time": time.time(),
                "detail": detail,
            }
        )
        self._write_payload()

    def _write_payload(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2))

    def _ensure_interaction_result_publisher(self) -> None:
        if self._interaction_result_pub is not None:
            return
        try:
            import rospy
            from std_msgs.msg import String

            publisher = rospy.Publisher(
                "/semantic_mapping/interaction_result",
                String,
                queue_size=2,
                latch=True,
            )
            self._interaction_result_pub = (publisher, String)
        except Exception as exc:
            log.warning("Door OCC test could not create interaction-result publisher: %s", exc)

    def _publish_interaction_result(self, detail, event_id: str, action: str) -> None:
        if self._interaction_result_pub is None:
            return
        publisher, message_type = self._interaction_result_pub
        joint_infos = [
            {
                "joint_name": transition["joint_name"],
                "joint_type": "hinge",
                "joint_range": list(transition["joint_range"]),
                "joint_value": float(transition["target"]),
            }
            for transition in detail.get("leaf_transitions") or []
        ]
        payload = {
            "event_id": str(event_id),
            "source_object_name": self.target_root_name,
            "state": str(detail.get("state") or self.current_state),
            "action": str(action),
            "joint_infos": joint_infos,
            "success": True,
            "confidence": 1.0,
            "source": "door_occ_direct_joint_readback",
            "verification_source": "mujoco_joint_readback",
            "stamp_sec": time.time(),
        }
        publisher.publish(message_type(data=json.dumps(payload, separators=(",", ":"))))


class DoorOccPoseSequenceController(DoorOccRuntimeController):
    """Teleport the robot through scheduled poses while opening and re-closing one door root."""

    def __init__(self, root_name: str, pose_sequence: str, output_path: str | Path):
        self.phases = parse_pose_sequence(pose_sequence)
        super().__init__(
            root_name=root_name,
            open_step=max(phase["step"] for phase in self.phases),
            output_path=output_path,
        )
        self.current_phase_index = -1
        self.payload.update(
            {
                "mode": "pose_sequence",
                "requested_phases": self.phases,
            }
        )

    def prepare(self, task) -> None:
        super().prepare(task)
        self._apply_phase(task.env, self.phases[0], record_step=0)
        self.current_phase_index = 0

    def before_step(self, task, step: int) -> bool:
        desired_index = max(
            index for index, phase in enumerate(self.phases) if int(phase["step"]) <= int(step)
        )
        phase = self.phases[desired_index]
        if desired_index != self.current_phase_index:
            self._apply_phase(task.env, phase, record_step=int(step))
            self.current_phase_index = desired_index
            return True

        door_detail = self._set_root_state(task.env, self.target_root_id, phase["state"])
        pose_changed, _pose_detail = self._set_robot_xyyaw(task.env, phase["robot_xyyaw"])
        if int(step) % 10 == 0:
            self._publish_interaction_result(
                door_detail,
                event_id=f"door_occ_pose_phase_{int(phase['phase_index']):02d}",
                action=phase["state"],
            )
        return bool(door_detail.get("joint_values_changed")) or pose_changed

    def _apply_phase(self, env, phase: dict[str, Any], record_step: int) -> None:
        door_detail = self._set_root_state(env, self.target_root_id, phase["state"])
        _pose_changed, pose_detail = self._set_robot_xyyaw(env, phase["robot_xyyaw"])
        if env.check_robot_collision_in_current_pose():
            raise RuntimeError(
                f"Door OCC pose phase {phase['label']!r} places the robot in collision at "
                f"{phase['robot_xyyaw']}"
            )
        self.current_state = phase["state"]
        transition = {
            "step": int(record_step),
            "phase_index": int(phase["phase_index"]),
            "label": str(phase["label"]),
            "state": str(phase["state"]),
            "robot_xyyaw": list(phase["robot_xyyaw"]),
            "wall_time": time.time(),
            "door_detail": door_detail,
            "pose_detail": pose_detail,
        }
        self.payload["transitions"].append(transition)
        self._write_payload()
        self._publish_interaction_result(
            door_detail,
            event_id=f"door_occ_pose_phase_{int(phase['phase_index']):02d}",
            action=phase["state"],
        )
        log.info(
            "Door OCC pose phase: index=%d step=%d label=%s state=%s robot_xyyaw=%s",
            int(phase["phase_index"]),
            int(record_step),
            phase["label"],
            phase["state"],
            phase["robot_xyyaw"],
        )

    @staticmethod
    def _set_robot_xyyaw(env, xyyaw) -> tuple[bool, dict[str, Any]]:
        x, y, yaw = (float(value) for value in xyyaw)
        robot_view = env.current_robot.robot_view
        before_pose = robot_view.base.pose.copy()
        before_yaw = math.atan2(float(before_pose[1, 0]), float(before_pose[0, 0]))
        target_pose = np.eye(4, dtype=float)
        target_pose[:3, :3] = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target_pose[:3, 3] = [x, y, float(before_pose[2, 3])]
        changed = (
            abs(float(before_pose[0, 3]) - x) > 1e-6
            or abs(float(before_pose[1, 3]) - y) > 1e-6
            or abs(math.atan2(math.sin(before_yaw - yaw), math.cos(before_yaw - yaw))) > 1e-6
        )
        if changed:
            robot_view.base.pose = target_pose
            env.current_data.qvel[:] = 0.0
            mujoco.mj_forward(env.current_model, env.current_data)
        return changed, {
            "before_xyyaw": [
                float(before_pose[0, 3]),
                float(before_pose[1, 3]),
                float(before_yaw),
            ],
            "target_xyyaw": [x, y, yaw],
            "changed": bool(changed),
        }
