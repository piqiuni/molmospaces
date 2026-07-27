from __future__ import annotations

import json
import math
import queue
from pathlib import Path
import time
from typing import Any

from force_interaction_runtime import (
    ForceDriveConfig,
    HeadViewController,
    advance_articulation_force,
    articulation_joint_infos,
    complete_articulation_force,
    collect_articulation_groups,
    finalize_articulation_force_transition,
    prepare_articulation_force,
    prepare_articulation_state_force,
    open_door_root_with_force,
    set_all_articulations_closed,
    set_all_door_roots_closed,
)


def ground_drawer_open_regions(
    joints: list[dict[str, Any]],
    open_regions: list[dict[str, Any]],
    body_heights: dict[str, float],
) -> list[dict[str, Any]]:
    """Match crop-space drawer centers to simulator-only slide joints by vertical order."""
    drawers = [
        dict(joint)
        for joint in joints
        if str(joint.get("joint_type") or "").casefold() == "slide"
        and str(joint.get("joint_name") or "")
    ]
    drawers.sort(
        key=lambda joint: (
            -float(body_heights.get(str(joint.get("joint_name") or ""), 0.0)),
            str(joint.get("joint_name") or ""),
        )
    )
    if not drawers:
        return []
    regions = [
        dict(region)
        for region in open_regions
        if isinstance(region, dict)
        and isinstance(region.get("center"), (list, tuple))
        and len(region["center"]) >= 2
    ]
    regions.sort(key=lambda region: (float(region["center"][1]), float(region["center"][0])))
    if not regions:
        return [
            {
                "group_id": f"sim_drawer_{index:02d}",
                "joint_names": [str(joint["joint_name"])],
                "open_region": None,
                "grounding_source": "simulator_all_slide_joints",
            }
            for index, joint in enumerate(drawers)
        ]

    available = list(range(len(drawers)))
    grounded = []
    for region_index, region in enumerate(regions[: len(drawers)]):
        y = max(0.0, min(1.0, float(region["center"][1])))
        target_rank = y * max(0, len(drawers) - 1)
        drawer_index = min(available, key=lambda index: (abs(index - target_rank), index))
        available.remove(drawer_index)
        grounded.append(
            {
                "group_id": f"visual_drawer_{region_index:02d}",
                "joint_names": [str(drawers[drawer_index]["joint_name"])],
                "open_region": region,
                "grounding_source": "visual_region_vertical_order",
                "simulator_rank": drawer_index,
            }
        )
    grounded.sort(key=lambda group: float((group["open_region"] or {})["center"][1]))
    return grounded


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
        interaction_execution_mode: str | None = None,
        interaction_transition_steps: int | None = None,
        drawer_execution_mode: str | None = None,
        drawer_transition_steps: int | None = None,
        drawer_observation_steps: int = 1,
    ) -> None:
        self.command_topic = str(command_topic)
        self.result_topic = str(result_topic)
        self.feedback_topic = str(feedback_topic)
        self.output_path = Path(output_path).expanduser().resolve() if output_path else None
        self.force_config = force_config or ForceDriveConfig()
        self.close_all_doors_on_prepare = bool(close_all_doors_on_prepare)
        self.close_all_containers_on_prepare = bool(close_all_containers_on_prepare)
        requested_mode = interaction_execution_mode or drawer_execution_mode or "fast"
        self.interaction_execution_mode = str(requested_mode).strip().lower()
        if self.interaction_execution_mode not in {"fast", "smooth"}:
            raise ValueError(
                f"Unsupported interaction execution mode: {requested_mode}"
            )
        requested_steps = interaction_transition_steps or drawer_transition_steps or 5
        self.interaction_transition_steps = max(1, int(requested_steps))
        self.drawer_execution_mode = self.interaction_execution_mode
        self.drawer_transition_steps = self.interaction_transition_steps
        self.drawer_observation_steps = max(1, int(drawer_observation_steps))
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
        self._force_observation_requested = False
        self._restore_view_pending = False
        self._head_view_controller = HeadViewController()
        self._last_view_restore_result: dict[str, Any] | None = None
        self._pause_navigation = False

    def prepare(self, task) -> None:
        self._ensure_ros()
        self._commands = queue.Queue()
        self._seen_command_ids.clear()
        self._event_index = 1
        self._events = []
        self._completed_steps = 0
        self._pending = None
        self._force_observation_requested = False
        self._restore_view_pending = False
        self._head_view_controller.reset()
        self._last_view_restore_result = None
        self._pause_navigation = False
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
        sequence_type = str(command.get("sequence_type") or "")
        if action != "open" and not (
            sequence_type == "drawer_scan" and action == "scan"
        ):
            raise ValueError(f"Atomic force interaction currently supports open only: {action}")
        object_id = str(command.get("object_id") or "")
        if not object_id:
            raise ValueError("Interaction command requires object_id")
        if not command_id:
            command_id = f"interaction_command_{self._event_index:06d}"
            command["command_id"] = command_id
        self._seen_command_ids.add(command_id)
        command["action"] = action
        command["object_id"] = object_id
        for planner_forbidden_key in (
            "source_object_name",
            "target_root_name",
            "root_body_name",
            "interaction_groups",
            "joint_names",
            "close_other_joint_names",
            "close_other_joints",
            "open_fraction_threshold",
        ):
            command.pop(planner_forbidden_key, None)
        self._commands.put(command)
        return True

    def before_step(self, task, step: int) -> dict[str, Any] | None:
        if self._pending is not None:
            if self._pending.get("kind") == "drawer_sequence":
                self._advance_drawer_sequence_before_step(task)
            elif self._pending.get("kind") == "articulation_sequence":
                self._advance_articulation_sequence_before_step(task, step)
            return None
        if self._restore_view_pending:
            self._restore_view_pending = False
            self._pause_navigation = True
            self._last_view_restore_result = self._head_view_controller.restore(task.env)
            if self._events:
                self._events[-1]["result"]["view_restore_result"] = (
                    self._last_view_restore_result
                )
                self._write_snapshot()
            self._force_observation_requested = True
            return None
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return None
        try:
            command["interaction_pose_validation"] = self._validate_interaction_pose(
                task, command
            )
            if str(command.get("sequence_type") or "") == "drawer_scan":
                self._start_drawer_sequence(task, command, step)
                self._advance_drawer_sequence_before_step(task)
                return None
            view_profile = str(command.get("view_profile") or "default")
            try:
                view_result = self._head_view_controller.command(
                    task.env,
                    view_profile,
                    tilt_rad=float(command.get("view_tilt_rad", 0.55) or 0.55),
                    torso_pitch_rad=command.get("view_torso_pitch_rad"),
                )
            except AttributeError:
                view_result = {
                    "profile": view_profile,
                    "applied": False,
                    "reason": "environment_view_profile_unavailable",
                }
            plan = prepare_articulation_force(
                task.env,
                command["object_id"],
            )
            if not bool(plan.get("supported", True)):
                return self._publish_unsupported_interaction(
                    command,
                    step=step,
                    reason=str(plan.get("reason") or "interaction_unsupported"),
                    interaction_capability=str(
                        plan.get("interaction_capability") or "unsupported"
                    ),
                    view_result=view_result,
                )
            execution_mode = str(
                command.get("interaction_execution_mode")
                or self.interaction_execution_mode
            ).strip().lower()
            if execution_mode not in {"fast", "smooth"}:
                raise ValueError(f"Unsupported interaction execution mode: {execution_mode}")
            self._pending = {
                "command": command,
                "plan": plan,
                "view_result": view_result,
                "step": int(step),
                "phase": "pre_interaction_hold",
                "remaining_view_hold_steps": max(
                    0, int(command.get("view_hold_task_steps", 0) or 0)
                ),
                "remaining_post_interaction_hold_steps": max(
                    0, int(command.get("post_interaction_hold_task_steps", 0) or 0)
                ),
                "execution_mode": execution_mode,
                "transition_steps": max(
                    1,
                    int(
                        command.get(
                            "interaction_transition_steps",
                            self.interaction_transition_steps,
                        )
                    ),
                ),
            }
            if execution_mode == "smooth":
                self._pending.update(
                    {
                        "kind": "articulation_sequence",
                        "phase": (
                            "pre_interaction_hold"
                            if self._pending["remaining_view_hold_steps"] > 0
                            else "interaction_transition"
                        ),
                        "transition_step": 0,
                        "transition_start_values": {
                            str(info["joint_name"]): float(info["joint_value"])
                            for info in plan["pre_joint_infos"]
                            if str(info.get("joint_name") or "") in plan["targets"]
                        },
                        "transition_log": [],
                        "physics_substeps": 0,
                        "atomic_fallback": False,
                    }
                )
                self._advance_articulation_sequence_before_step(task, step)
            self._pause_navigation = True
            if view_profile == "drawer_low_view":
                self._force_observation_requested = True
            return None
        except (KeyError, ValueError) as exc:
            self._commands.task_done()
            return self._publish_command_failure(
                task,
                command,
                step,
                exc,
                view_result=locals().get("view_result"),
            )
        except Exception:
            self._commands.task_done()
            raise

    def _publish_unsupported_interaction(
        self,
        command: dict[str, Any],
        *,
        step: int,
        reason: str,
        interaction_capability: str,
        view_result: dict[str, Any],
    ) -> dict[str, Any]:
        event_id = str(
            command.get("event_id") or f"interaction_{self._event_index:06d}"
        )
        self._event_index += 1
        stamp_sec = time.time()
        node_type = str(command.get("node_type") or "").casefold()
        static_portal = (
            node_type == "portal"
            and str(interaction_capability).casefold() == "static"
        )
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": str(command.get("node_id") or ""),
            "object_id": str(command["object_id"]),
            "node_type": node_type,
            "action": str(command.get("action") or "open"),
            "interaction_mode": "none",
            "interaction_group_id": str(
                command.get("interaction_group_id") or "all"
            ),
            "interaction_capability": str(interaction_capability),
            "interactable": False,
            "state": "static_open" if static_portal else "static",
            "pre_state": "unknown",
            "post_state": "static_open" if static_portal else "unknown",
            "success": static_portal,
            "status": "SUCCEEDED" if static_portal else "FAILED",
            "reason": "" if static_portal else str(reason),
            "confidence": 1.0,
            "execution_cost": 0.0,
            "sim_steps_consumed": 0,
            "physics_substeps": 0,
            "task_steps_consumed": 0,
            "result_published_step": int(step),
            "source": (
                "executor_static_portal"
                if static_portal
                else "force_interaction_capability_check"
            ),
            "verification_source": "mujoco_articulation_registry",
            "view_profile": str(command.get("view_profile") or "default"),
            "view_profile_result": view_result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        feedback = {
            "command_id": result["command_id"],
            "candidate_id": result["candidate_id"],
            "decision_id": result["decision_id"],
            "event_id": event_id,
            "behavior_type": "INTERACT",
            "status": result["status"],
            "success": result["success"],
            "reason": result["reason"],
            "interaction_result": result,
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._force_observation_requested = True
        self._pause_navigation = False
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        self._commands.task_done()
        return result

    def after_step(self, task, step: int) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None:
            return None
        if pending.get("kind") == "drawer_sequence":
            return self._advance_drawer_sequence_after_step(task, step)
        if pending.get("phase") == "pre_interaction_hold" and int(
            pending.get("remaining_view_hold_steps", 0)
        ) > 0:
            pending["remaining_view_hold_steps"] -= 1
            self._force_observation_requested = True
            if (
                pending.get("kind") == "articulation_sequence"
                and int(pending["remaining_view_hold_steps"]) <= 0
            ):
                pending["phase"] = "interaction_transition"
            return None
        command = pending["command"]
        if (
            pending.get("kind") == "articulation_sequence"
            and pending.get("phase") == "interaction_transition"
        ):
            if int(pending["transition_step"]) < int(pending["transition_steps"]):
                return None
            force_result = finalize_articulation_force_transition(
                task.env,
                pending["plan"],
                physics_substeps=int(pending["physics_substeps"]),
                task_steps_consumed=int(pending["transition_steps"]),
                atomic_fallback=bool(pending["atomic_fallback"]),
                transition_log=list(pending["transition_log"]),
                config=self.force_config,
            )
            pending["force_result"] = force_result
            pending["phase"] = "post_interaction_hold"
            self._force_observation_requested = True
            if int(pending.get("remaining_post_interaction_hold_steps", 0)) > 0:
                return None
        elif pending.get("phase") == "pre_interaction_hold":
            force_result = complete_articulation_force(
                task.env,
                pending["plan"],
                config=self.force_config,
            )
            pending["force_result"] = force_result
            pending["force_applied_step"] = int(step)
            pending["phase"] = "post_interaction_hold"
            self._force_observation_requested = True
            if int(pending.get("remaining_post_interaction_hold_steps", 0)) > 0:
                return None
        if pending.get("phase") == "post_interaction_hold" and int(
            pending.get("remaining_post_interaction_hold_steps", 0)
        ) > 0:
            pending["remaining_post_interaction_hold_steps"] -= 1
            self._force_observation_requested = True
            if int(pending["remaining_post_interaction_hold_steps"]) > 0:
                return None
        self._pending = None
        try:
            force_result = pending["force_result"]
            event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
            self._event_index += 1
            stamp_sec = time.time()
            result = {
                "event_id": event_id,
                "command_id": str(command["command_id"]),
                "candidate_id": str(command.get("candidate_id") or ""),
                "decision_id": str(command.get("decision_id") or ""),
                "node_id": str(command.get("node_id") or ""),
                "object_id": str(command["object_id"]),
                "action": "open",
                "interaction_mode": str(command.get("interaction_mode") or "open_close"),
                "operation_method": str(command.get("operation_method") or "unknown"),
                "open_regions": list(command.get("open_regions") or []),
                "approach_goal_xyyaw": list(
                    command.get("approach_goal_xyyaw") or []
                ),
                "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
                "view_profile": str(command.get("view_profile") or "default"),
                "view_torso_pitch_rad": command.get("view_torso_pitch_rad"),
                "view_hold_task_steps": int(
                    command.get("view_hold_task_steps", 0) or 0
                ),
                "post_interaction_hold_task_steps": int(
                    command.get("post_interaction_hold_task_steps", 0) or 0
                ),
                "view_profile_result": pending["view_result"],
                "view_restore_result": self._last_view_restore_result,
                "method": "xfrc_applied_group_pd",
                "state": str(force_result["post_state"]),
                "pre_state": str(force_result["pre_state"]),
                "post_state": str(force_result["post_state"]),
                "success": bool(force_result["success"]),
                "status": "SUCCEEDED" if force_result["success"] else "FAILED",
                "confidence": 1.0,
                "execution_cost": 1.0,
                "sim_steps_consumed": int(step) - int(pending["step"]) + 1,
                "physics_substeps": int(force_result["physics_substeps"]),
                "task_steps_consumed": int(force_result["task_steps_consumed"]),
                "force_applied_step": int(
                    pending.get("force_applied_step", pending["step"])
                ),
                "result_published_step": int(step),
                "interaction_execution_mode": str(
                    pending.get("execution_mode") or "fast"
                ),
                "interaction_transition_steps": int(
                    pending.get("transition_steps") or 1
                ),
                "source": (
                    "force_smooth_interaction"
                    if pending.get("execution_mode") == "smooth"
                    else "force_atomic_interaction"
                ),
                "verification_source": "executor_state_verification",
                "interaction_pose_validation": dict(
                    command.get("interaction_pose_validation") or {}
                ),
                "step": int(pending["step"]),
                "stamp_sec": stamp_sec,
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
            self._force_observation_requested = True
            self._restore_view_pending = bool(
                command.get("restore_view_after", False)
            )
            if self._restore_view_pending:
                self._last_view_restore_result = None
            self._pause_navigation = self._restore_view_pending
            self._publish(self._result_publisher, result)
            self._publish(self._feedback_publisher, feedback)
            self._write_snapshot()
            return result
        finally:
            self._commands.task_done()

    def _advance_articulation_sequence_before_step(self, task, step: int) -> None:
        pending = self._pending
        if (
            pending is None
            or pending.get("kind") != "articulation_sequence"
            or pending.get("phase") != "interaction_transition"
        ):
            return
        transition_steps = int(pending["transition_steps"])
        if int(pending["transition_step"]) >= transition_steps:
            return
        next_step = int(pending["transition_step"]) + 1
        transition = advance_articulation_force(
            task.env,
            pending["plan"],
            progress=float(next_step) / float(transition_steps),
            start_values=pending["transition_start_values"],
            transition_steps=transition_steps,
            config=self.force_config,
        )
        pending["transition_step"] = next_step
        pending["physics_substeps"] += int(transition.get("physics_substeps", 0))
        pending["atomic_fallback"] = bool(
            pending["atomic_fallback"] or transition.get("fallback", False)
        )
        pending["transition_log"].append(
            {
                "task_step": int(step),
                "task_step_index": next_step,
                "progress": float(transition.get("progress", next_step / transition_steps)),
                "fallback": bool(transition.get("fallback", False)),
                "physics_substeps": int(transition.get("physics_substeps", 0)),
            }
        )
        pending.setdefault("force_applied_step", int(step))
        self._force_observation_requested = True

    def _start_drawer_sequence(self, task, command: dict[str, Any], step: int) -> None:
        articulation_groups = collect_articulation_groups(task.env)
        articulation = articulation_groups.get(str(command["object_id"]))
        if articulation is None:
            raise ValueError(
                "drawer_scan target is not an articulated simulator object: "
                f"{command['object_id']}"
            )
        body_heights = {}
        model = task.env.current_model
        data = task.env.current_data
        for joint in articulation.get("joints") or []:
            joint_name = str(joint.get("joint_name") or "")
            if not joint_name:
                continue
            body_id = None
            body_name = str(joint.get("body_name") or "")
            if body_name:
                try:
                    body_id = int(model.body(body_name).id)
                except (KeyError, ValueError):
                    body_id = None
            if body_id is None and joint.get("joint_id") is not None:
                body_id = int(model.jnt_bodyid[int(joint["joint_id"])])
            if body_id is not None:
                body_heights[joint_name] = float(data.xpos[body_id][2])
        groups = ground_drawer_open_regions(
            list(articulation.get("joints") or []),
            list(command.get("open_regions") or []),
            body_heights,
        )
        if not groups:
            raise ValueError("drawer_scan requires at least one slide joint")
        self._pending = {
            "kind": "drawer_sequence",
            "command": command,
            "step": int(step),
            "phase": "open",
            "phase_step": 0,
            "group_index": 0,
            "groups": groups,
            "all_joint_names": [
                name for group in groups for name in group["joint_names"]
            ],
            "transition_steps": max(
                1,
                int(command.get("drawer_transition_steps", self.drawer_transition_steps)),
            ),
            "observation_steps": max(
                1,
                int(command.get("drawer_observation_steps", self.drawer_observation_steps)),
            ),
            "remaining_observation_steps": 0,
            "phase_plan": None,
            "phase_start_values": {},
            "group_results": [],
            "transition_log": [],
            "physics_substeps": 0,
            "view_result": None,
            "view_restore_result": None,
        }
        self._pause_navigation = True

    def _advance_drawer_sequence_before_step(self, task) -> None:
        pending = self._pending
        if pending is None or pending.get("kind") != "drawer_sequence":
            return
        phase = str(pending["phase"])
        if phase == "observe":
            return
        groups = pending["groups"]
        group_index = int(pending["group_index"])
        current_group = groups[group_index]
        mode = self.interaction_execution_mode
        transition_steps = int(pending["transition_steps"]) if mode == "smooth" else 1
        if pending.get("phase_plan") is None:
            if phase == "open":
                close_names = [
                    name
                    for name in pending["all_joint_names"]
                    if name not in current_group["joint_names"]
                ]
                pending["phase_plan"] = prepare_articulation_state_force(
                    task.env,
                    pending["command"]["object_id"],
                    open_joint_names=current_group["joint_names"],
                    close_joint_names=close_names,
                )
            else:
                pending["phase_plan"] = prepare_articulation_state_force(
                    task.env,
                    pending["command"]["object_id"],
                    close_joint_names=pending["all_joint_names"],
                )
            pending["phase_start_values"] = {
                str(info["joint_name"]): float(info["joint_value"])
                for info in pending["phase_plan"]["pre_joint_infos"]
                if str(info.get("joint_name") or "")
                in pending["phase_plan"]["targets"]
            }
        next_step = int(pending["phase_step"]) + 1
        progress = min(1.0, float(next_step) / float(transition_steps))
        command = pending["command"]
        if phase == "open" and group_index == 0:
            view_progress = progress if mode == "smooth" else 1.0
            view_result = self._head_view_controller.command(
                task.env,
                "drawer_low_view",
                tilt_rad=float(command.get("view_tilt_rad", 0.30) or 0.30)
                * view_progress,
                torso_pitch_rad=float(
                    command.get("view_torso_pitch_rad", 0.35) or 0.35
                )
                * view_progress,
            )
            if pending["view_result"] is None:
                pending["view_result"] = view_result
        if phase == "close" and group_index == len(groups) - 1:
            view_progress = progress if mode == "smooth" else 1.0
            self._head_view_controller.command(
                task.env,
                "drawer_low_view",
                tilt_rad=float(command.get("view_tilt_rad", 0.30) or 0.30)
                * (1.0 - view_progress),
                torso_pitch_rad=float(
                    command.get("view_torso_pitch_rad", 0.35) or 0.35
                )
                * (1.0 - view_progress),
            )
        transition = advance_articulation_force(
            task.env,
            pending["phase_plan"],
            progress=progress,
            start_values=pending["phase_start_values"],
            transition_steps=transition_steps,
            config=self.force_config,
        )
        pending["phase_step"] = next_step
        pending["physics_substeps"] += int(transition.get("physics_substeps", 0))
        pending["transition_log"].append(
            {
                "phase": phase,
                "group_id": current_group["group_id"],
                "task_step_index": next_step,
                "progress": progress,
                "fallback": bool(transition.get("fallback", False)),
                "physics_substeps": int(transition.get("physics_substeps", 0)),
            }
        )

    def _advance_drawer_sequence_after_step(
        self, task, step: int
    ) -> dict[str, Any] | None:
        pending = self._pending
        if pending is None:
            return None
        phase = str(pending["phase"])
        mode = self.interaction_execution_mode
        transition_steps = int(pending["transition_steps"]) if mode == "smooth" else 1
        if phase in {"open", "close"} and int(pending["phase_step"]) < transition_steps:
            return None
        if phase == "open":
            if mode == "fast":
                self._force_observation_requested = True
                if int(pending["observation_steps"]) > 1:
                    pending["phase"] = "observe"
                    pending["phase_step"] = 0
                    pending["phase_plan"] = None
                    pending["remaining_observation_steps"] = int(
                        pending["observation_steps"]
                    )
                    return None
                self._record_drawer_observation(task, step)
                if int(pending["group_index"]) + 1 < len(pending["groups"]):
                    pending["group_index"] += 1
                    pending["phase"] = "open"
                else:
                    pending["phase"] = "close"
                pending["phase_step"] = 0
                pending["phase_plan"] = None
                return None
            pending["phase"] = "observe"
            pending["phase_step"] = 0
            pending["phase_plan"] = None
            pending["remaining_observation_steps"] = int(pending["observation_steps"])
            return None
        if phase == "observe":
            self._force_observation_requested = True
            pending["remaining_observation_steps"] -= 1
            if int(pending["remaining_observation_steps"]) > 0:
                return None
            self._record_drawer_observation(task, step)
            pending["phase"] = "close"
            pending["phase_step"] = 0
            pending["phase_plan"] = None
            return None
        if phase == "close":
            if int(pending["group_index"]) + 1 < len(pending["groups"]):
                pending["group_index"] += 1
                pending["phase"] = "open"
                pending["phase_step"] = 0
                pending["phase_plan"] = None
                return None
            pending["view_restore_result"] = self._head_view_controller.restore(task.env)
            return self._finish_drawer_sequence(task, step)
        return None

    def _record_drawer_observation(self, task, step: int) -> None:
        pending = self._pending
        group = pending["groups"][int(pending["group_index"])]
        joint_infos = articulation_joint_infos(
            task.env, pending["command"]["object_id"]
        )
        selected_infos = [
            info for info in joint_infos if info["joint_name"] in group["joint_names"]
        ]
        success = bool(selected_infos) and all(
            float(info.get("open_fraction", 0.0))
            >= float(self.force_config.open_fraction_threshold)
            for info in selected_infos
        )
        pending["group_results"].append(
            {
                "region_id": group["group_id"],
                "open_region": group.get("open_region"),
                "grounding_source": group.get(
                    "grounding_source", "simulator_articulation_mapping"
                ),
                "success": success,
                "observation_step": int(step),
            }
        )

    def _finish_drawer_sequence(self, task, step: int) -> dict[str, Any]:
        pending = self._pending
        command = pending["command"]
        final_joint_infos = articulation_joint_infos(
            task.env, command["object_id"]
        )
        success = bool(pending["group_results"]) and all(
            bool(group.get("success")) for group in pending["group_results"]
        )
        final_close_success = bool(final_joint_infos) and all(
            float(info.get("open_fraction", 1.0))
            <= 1.0 - float(self.force_config.open_fraction_threshold)
            for info in final_joint_infos
            if info.get("joint_name") in pending["all_joint_names"]
        )
        success = success and final_close_success
        event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
        self._event_index += 1
        stamp_sec = time.time()
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": str(command.get("node_id") or ""),
            "object_id": str(command["object_id"]),
            "action": "scan",
            "interaction_mode": "drawer_scan",
            "sequence_type": "drawer_scan",
            "operation_method": str(command.get("operation_method") or "pull"),
            "open_regions": list(command.get("open_regions") or []),
            "approach_goal_xyyaw": list(command.get("approach_goal_xyyaw") or []),
            "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
            "grounded_regions": [
                {
                    "region_id": str(group.get("group_id") or ""),
                    "open_region": group.get("open_region"),
                    "grounding_source": str(
                        group.get("grounding_source")
                        or "simulator_articulation_mapping"
                    ),
                }
                for group in pending["groups"]
            ],
            "region_results": list(pending["group_results"]),
            "final_close_success": final_close_success,
            "view_profile": "drawer_low_view",
            "view_profile_result": pending["view_result"],
            "view_restore_result": pending["view_restore_result"],
            "interaction_execution_mode": self.interaction_execution_mode,
            "interaction_transition_steps": int(pending["transition_steps"]),
            "drawer_execution_mode": self.interaction_execution_mode,
            "drawer_transition_steps": int(pending["transition_steps"]),
            "drawer_observation_steps": int(pending["observation_steps"]),
            "transition_log": list(pending["transition_log"]),
            "state": "closed",
            "pre_state": "closed",
            "post_state": "closed",
            "success": success,
            "status": "SUCCEEDED" if success else "FAILED",
            "confidence": 1.0,
            "execution_cost": 1.0,
            "sim_steps_consumed": int(step) - int(pending["step"]) + 1,
            "physics_substeps": int(pending["physics_substeps"]),
            "task_steps_consumed": int(step) - int(pending["step"]) + 1,
            "result_published_step": int(step),
            "source": "force_container_sequence",
            "verification_source": "executor_state_verification",
            "step": int(pending["step"]),
            "stamp_sec": stamp_sec,
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
        self._pending = None
        self._pause_navigation = False
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        self._commands.task_done()
        return result

    def consume_force_observation_request(self) -> bool:
        requested = self._force_observation_requested
        self._force_observation_requested = False
        return requested

    def should_pause_navigation(self) -> bool:
        return self._pending is not None or self._pause_navigation

    def view_torso_target(self) -> list[float] | None:
        return self._head_view_controller.torso_target()

    def after_task_step(self) -> None:
        if self._pending is None and not self._restore_view_pending:
            self._pause_navigation = False

    @staticmethod
    def _validate_interaction_pose(task, command: dict[str, Any]) -> dict[str, Any]:
        expected = list(command.get("interaction_approach_pose_xyyaw") or [])
        if len(expected) < 3:
            return {"checked": False, "reason": "no_expected_approach_pose"}
        base_pose = task.env.current_robot.robot_view.base.pose
        actual = [
            float(base_pose[0, 3]),
            float(base_pose[1, 3]),
            math.atan2(float(base_pose[1, 0]), float(base_pose[0, 0])),
        ]
        position_error_m = math.hypot(
            actual[0] - float(expected[0]), actual[1] - float(expected[1])
        )
        yaw_error_rad = abs(
            math.atan2(
                math.sin(actual[2] - float(expected[2])),
                math.cos(actual[2] - float(expected[2])),
            )
        )
        distance_tolerance_m = max(
            0.05, float(command.get("interaction_ready_distance_m", 0.45) or 0.45)
        )
        yaw_tolerance_rad = max(
            0.05,
            float(
                command.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
            ),
        )
        valid = (
            position_error_m <= distance_tolerance_m
            and yaw_error_rad <= yaw_tolerance_rad
        )
        result = {
            "checked": True,
            "valid": valid,
            "expected_pose_xyyaw": [float(value) for value in expected[:3]],
            "actual_pose_xyyaw": actual,
            "position_error_m": position_error_m,
            "yaw_error_rad": yaw_error_rad,
            "distance_tolerance_m": distance_tolerance_m,
            "yaw_tolerance_rad": yaw_tolerance_rad,
            "approach_axis_xy": list(
                command.get("interaction_approach_axis_xy") or []
            ),
        }
        if not valid:
            command["interaction_pose_validation"] = result
            raise ValueError(
                "Interaction pose invalid: "
                f"position_error_m={position_error_m:.3f} "
                f"yaw_error_rad={yaw_error_rad:.3f}"
            )
        return result

    def _publish_command_failure(
        self,
        task,
        command: dict[str, Any],
        step: int,
        exc: Exception,
        view_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = str(command.get("event_id") or f"interaction_{self._event_index:06d}")
        self._event_index += 1
        stamp_sec = time.time()
        object_id = str(command.get("object_id") or "")
        node_id = str(command.get("node_id") or "")
        node_type = str(command.get("node_type") or "").casefold()
        missing_articulation = str(exc).startswith("Articulated object not found:")
        invalid_interaction_pose = str(exc).startswith("Interaction pose invalid:")
        static_portal = missing_articulation and node_type == "portal"
        try:
            view_restore_result = self._head_view_controller.restore(task.env)
        except (AttributeError, KeyError, ValueError):
            view_restore_result = {
                "applied": False,
                "reason": "environment_view_restore_unavailable",
            }
        result = {
            "event_id": event_id,
            "command_id": str(command["command_id"]),
            "candidate_id": str(command.get("candidate_id") or ""),
            "decision_id": str(command.get("decision_id") or ""),
            "node_id": node_id,
            "object_id": object_id,
            "node_type": node_type,
            "action": str(command.get("action") or "open"),
            "interaction_mode": str(command.get("interaction_mode") or "open_close"),
            "operation_method": str(command.get("operation_method") or "unknown"),
            "open_regions": list(command.get("open_regions") or []),
            "visual_operation_plan": dict(command.get("visual_operation_plan") or {}),
            "view_profile": str(command.get("view_profile") or "default"),
            "view_profile_result": view_result,
            "view_restore_result": view_restore_result,
            "state": "static_open" if static_portal else "unknown",
            "pre_state": "unknown",
            "post_state": "static_open" if static_portal else "unknown",
            "success": static_portal,
            "status": "SUCCEEDED" if static_portal else "FAILED",
            "confidence": 1.0,
            "execution_cost": 0.0 if static_portal else 1.0,
            "sim_steps_consumed": 0,
            "physics_substeps": 0,
            "task_steps_consumed": 0,
            "result_published_step": int(step),
            "interaction_execution_mode": self.interaction_execution_mode,
            "interaction_transition_steps": 0,
            "source": (
                "executor_static_portal"
                if static_portal
                else "force_interaction_rejected"
            ),
            "verification_source": (
                "simulator_no_articulation"
                if static_portal
                else (
                    "executor_pose_precondition"
                    if invalid_interaction_pose
                    else "executor_resolution_failure"
                )
            ),
            "failure_reason": (
                ""
                if static_portal
                else (
                    "interaction_pose_invalid"
                    if invalid_interaction_pose
                    else "articulation_resolution_failed"
                )
            ),
            "interaction_pose_validation": dict(
                command.get("interaction_pose_validation") or {}
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "step": int(step),
            "stamp_sec": stamp_sec,
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
            "step": int(step),
            "stamp_sec": stamp_sec,
        }
        self._events.append({"result": result, "feedback": feedback})
        self._pending = None
        self._restore_view_pending = False
        self._pause_navigation = False
        self._force_observation_requested = True
        self._publish(self._result_publisher, result)
        self._publish(self._feedback_publisher, feedback)
        self._write_snapshot()
        return result

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
