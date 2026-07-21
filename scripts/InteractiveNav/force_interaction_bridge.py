from __future__ import annotations

import json
import queue
from pathlib import Path
import time
from typing import Any

from force_interaction_runtime import (
    ForceDriveConfig,
    apply_view_profile,
    complete_articulation_force,
    drive_joint_group_to_targets,
    prepare_articulation_force,
    set_all_articulations_closed,
    set_all_door_roots_closed,
)


class AtomicForceInteractionController:
    def __init__(
        self,
        command_topic: str = "/semantic_decision/interaction_command",
        result_topic: str = "/semantic_mapping/interaction_result",
        feedback_topic: str = "/semantic_decision/interaction_action_feedback",
        output_path: str | Path | None = None,
        force_config: ForceDriveConfig | None = None,
        close_all_doors_on_prepare: bool = True,
        close_all_containers_on_prepare: bool = False,
    ) -> None:
        self.command_topic = str(command_topic)
        self.result_topic = str(result_topic)
        self.feedback_topic = str(feedback_topic)
        self.output_path = Path(output_path).expanduser().resolve() if output_path else None
        self.force_config = force_config or ForceDriveConfig()
        self.close_all_doors_on_prepare = bool(close_all_doors_on_prepare)
        self.close_all_containers_on_prepare = bool(close_all_containers_on_prepare)
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._seen_command_ids: set[str] = set()
        self._event_index = 1
        self._result_publisher = None
        self._feedback_publisher = None
        self._subscriber = None
        self._String = None
        self._initial_door_states: list[dict[str, Any]] = []
        self._initial_articulation_states: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._completed_steps = 0
        self._pending: dict[str, Any] | None = None

    def prepare(self, task) -> None:
        self._ensure_ros()
        self._commands = queue.Queue()
        self._seen_command_ids.clear()
        self._event_index = 1
        self._events = []
        self._completed_steps = 0
        self._pending = None
        if self.close_all_containers_on_prepare:
            self._initial_articulation_states = set_all_articulations_closed(
                task.env,
                include_doors=self.close_all_doors_on_prepare,
            )
            self._initial_door_states = [
                row for row in self._initial_articulation_states if row.get("is_door")
            ]
        else:
            self._initial_articulation_states = []
            self._initial_door_states = (
                set_all_door_roots_closed(task.env) if self.close_all_doors_on_prepare else []
            )
        self._write_snapshot()

    def enqueue_command(self, payload: dict[str, Any]) -> bool:
        command = dict(payload or {})
        command_id = str(command.get("command_id") or "")
        if command_id and command_id in self._seen_command_ids:
            return False
        action = str(command.get("action") or "open").lower()
        if action != "open":
            raise ValueError(f"Atomic force interaction currently supports open only: {action}")
        root_name = str(
            command.get("source_object_name")
            or command.get("target_root_name")
            or command.get("root_body_name")
            or ""
        )
        if not root_name:
            raise ValueError("Interaction command requires source_object_name or target_root_name")
        if not command_id:
            command_id = f"interaction_command_{self._event_index:06d}"
            command["command_id"] = command_id
        self._seen_command_ids.add(command_id)
        command["action"] = action
        command["source_object_name"] = root_name
        self._commands.put(command)
        return True

    def before_step(self, task, step: int) -> dict[str, Any] | None:
        if self._pending is not None:
            return None
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return None
        try:
            view_profile = str(command.get("view_profile") or "default")
            try:
                view_result = apply_view_profile(
                    task.env,
                    view_profile,
                    tilt_rad=float(command.get("view_tilt_rad", 0.55) or 0.55),
                )
            except AttributeError:
                view_result = {
                    "profile": view_profile,
                    "applied": False,
                    "reason": "environment_view_profile_unavailable",
                }
            plan = prepare_articulation_force(
                task.env,
                command["source_object_name"],
                selected_joint_names=list(command.get("joint_names") or []) or None,
                close_other_joint_names=(
                    list(command.get("close_other_joint_names") or []) or None
                ),
                close_other_joints=bool(command.get("close_other_joints", False)),
            )
            plan["drive"] = drive_joint_group_to_targets(
                task.env.current_model,
                task.env.current_data,
                plan["targets"],
                config=self.force_config,
            )
            self._pending = {
                "command": command,
                "plan": plan,
                "view_result": view_result,
                "step": int(step),
            }
            return None
        except Exception:
            self._commands.task_done()
            raise

    def after_step(self, task, step: int) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None:
            return None
        self._pending = None
        command = pending["command"]
        try:
            force_result = complete_articulation_force(
                task.env,
                pending["plan"],
                config=self.force_config,
            )
            event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
            self._event_index += 1
            stamp_sec = time.time()
            result = {
                "event_id": event_id,
                "command_id": str(command["command_id"]),
                "candidate_id": str(command.get("candidate_id") or ""),
                "decision_id": str(command.get("decision_id") or ""),
                "node_id": str(command.get("node_id") or ""),
                "source_object_name": str(command["source_object_name"]),
                "action": "open",
                "interaction_mode": str(command.get("interaction_mode") or "open_close"),
                "interaction_group_id": str(command.get("interaction_group_id") or "all"),
                "joint_names": list(force_result.get("selected_joint_names") or []),
                "close_other_joint_names": list(force_result.get("closed_joint_names") or []),
                "view_profile": str(command.get("view_profile") or "default"),
                "view_profile_result": pending["view_result"],
                "method": "xfrc_applied_group_pd",
                "state": str(force_result["post_state"]),
                "pre_state": str(force_result["pre_state"]),
                "post_state": str(force_result["post_state"]),
                "joint_infos": list(force_result["joint_infos"]),
                "success": bool(force_result["success"]),
                "status": "SUCCEEDED" if force_result["success"] else "FAILED",
                "confidence": 1.0,
                "execution_cost": 1.0,
                "sim_steps_consumed": 1,
                "physics_substeps": int(force_result["physics_substeps"]),
                "task_steps_consumed": int(force_result["task_steps_consumed"]),
                "source": "force_atomic_interaction",
                "verification_source": "mujoco_joint_readback",
                "step": int(pending["step"]),
                "stamp_sec": stamp_sec,
                "force_result": force_result,
            }
            feedback = {
                "command_id": result["command_id"],
                "candidate_id": result["candidate_id"],
                "decision_id": result["decision_id"],
                "event_id": event_id,
                "behavior_type": "INTERACT",
                "status": result["status"],
                "success": result["success"],
                "interaction_result": result,
                "step": int(pending["step"]),
                "stamp_sec": stamp_sec,
            }
            self._events.append({"result": result, "feedback": feedback})
            self._publish(self._result_publisher, result)
            self._publish(self._feedback_publisher, feedback)
            self._write_snapshot()
            return result
        finally:
            self._commands.task_done()

    def finalize(self, completed_steps: int) -> None:
        self._completed_steps = int(completed_steps)
        self._write_snapshot()

    def close(self) -> None:
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None

    def _command_callback(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("interaction command must be a JSON object")
            self.enqueue_command(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                import rospy

                rospy.logwarn("Atomic force interaction command ignored: %s", exc)
            except Exception:
                pass

    def _ensure_ros(self) -> None:
        if self._subscriber is not None:
            return
        import rospy
        from std_msgs.msg import String

        self._String = String
        self._result_publisher = rospy.Publisher(
            self.result_topic,
            String,
            queue_size=4,
            latch=True,
        )
        self._feedback_publisher = rospy.Publisher(
            self.feedback_topic,
            String,
            queue_size=4,
            latch=True,
        )
        self._subscriber = rospy.Subscriber(
            self.command_topic,
            String,
            self._command_callback,
            queue_size=8,
        )

    def _publish(self, publisher, payload: dict[str, Any]) -> None:
        if publisher is None or self._String is None:
            return
        publisher.publish(
            self._String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def _write_snapshot(self) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "command_topic": self.command_topic,
                    "result_topic": self.result_topic,
                    "feedback_topic": self.feedback_topic,
                    "force_config": self.force_config.__dict__,
                    "initial_door_states": self._initial_door_states,
                    "initial_articulation_states": self._initial_articulation_states,
                    "events": self._events,
                    "completed_steps": self._completed_steps,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
