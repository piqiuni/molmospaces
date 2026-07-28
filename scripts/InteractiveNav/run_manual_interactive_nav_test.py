from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.env.data_views import Door
from molmo_spaces.utils.pose import pos_quat_to_pose_mat
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav.manual_interactive_nav_camera import (
    ManualExocentricCameraController,
)
from scripts.InteractiveNav.manual_interactive_nav_policy import (
    ManualControlEvent,
    ManualInteractiveNavPolicy,
)


DEFAULT_CATALOG = REPO_ROOT / (
    "scripts/InteractiveNav/output/mixed_rough_catalog_occfix_all_strict_v2_20260718/"
    "mixed_rough_catalog.json"
)
DEFAULT_CASE_ID = (
    "mixed_h783__refrigerator_2c47eb19b983297ad27dd38b449e9660_1_0_2__"
    "egg_7331ece1e05e35ccf61e870ea8948aac_1_0_2"
)
DEFAULT_CAPTURE_DIR = REPO_ROOT / "scripts/InteractiveNav/output/manual_interactive_nav"
DEFAULT_CAMERA_TARGET = np.asarray([5.4835, 14.6857, 0.9], dtype=float)
DEFAULT_OVER_SHOULDER_POSITION = np.asarray([-1.8, -1.2, 1.85], dtype=float)
DEFAULT_OVER_SHOULDER_LOOKAT = np.asarray([1.4, 0.0, 1.0], dtype=float)
WINDOW_CONTINUOUS_KEYS = (
    ManualInteractiveNavPolicy.MOVEMENT_KEYS
    | ManualInteractiveNavPolicy.CAMERA_KEYS
)


@dataclass(frozen=True)
class InteractionTarget:
    kind: str
    name: str
    label: str
    distance_m: float
    record: dict[str, Any]
    joint_indices: tuple[int, ...] = ()


def load_candidate(catalog_path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with catalog_path.open() as handle:
        payload = json.load(handle)
    candidate = next(
        (row for row in payload["candidates"] if row["case_id"] == case_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Mixed rough candidate not found: {case_id}")
    return candidate, payload


def load_context(candidate: dict[str, Any], payload: dict[str, Any], output_dir: Path):
    episodes = container_builder.load_benchmark_episodes(Path(payload["benchmark_dir"]))
    episode_index = int(candidate["source_episode_index"])
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"Episode index {episode_index} is outside benchmark size {len(episodes)}")
    episode = episodes[episode_index]
    if int(episode["house_index"]) != int(candidate["house_index"]):
        raise ValueError(
            f"Episode {episode_index} is house {episode['house_index']}, expected "
            f"house {candidate['house_index']}"
        )
    load_args = argparse.Namespace(robot="rby1", variant="base", seed=0, output_dir=output_dir)
    return container_builder.load_episode_context(load_args, episode), episode


def aabb_xy_distance(robot_xy: np.ndarray, center: Any, size: Any) -> float:
    center_xy = np.asarray(center, dtype=float)[:2]
    half_size_xy = np.asarray(size, dtype=float)[:2] / 2.0
    delta = np.maximum(np.abs(np.asarray(robot_xy, dtype=float) - center_xy) - half_size_xy, 0.0)
    return float(np.linalg.norm(delta))


def interaction_targets(
    env,
    candidate: dict[str, Any],
    doorway_analysis: dict[str, Any],
    door_records: list[dict[str, Any]],
    containers: list[dict[str, Any]],
) -> list[InteractionTarget]:
    robot_xy = np.asarray(env.current_robot.robot_view.base.pose[:2, 3], dtype=float)
    targets: list[InteractionTarget] = []
    for record in door_records:
        distance = aabb_xy_distance(robot_xy, record["aabb_center"], record["aabb_size"])
        targets.append(
            InteractionTarget(
                kind="door",
                name=str(record["name"]),
                label=f"Door {record['name']}",
                distance_m=distance,
                record=record,
            )
        )
    for record in containers:
        joints = record.get("joints", [])
        if not joints:
            continue
        if record["name"] == candidate["container_name"]:
            requested = tuple(int(index) for index in candidate["joint_sequence"])
            available = {int(joint["joint_index"]) for joint in joints}
            joint_indices = tuple(index for index in requested if index in available)
        else:
            joint_indices = (int(joints[0]["joint_index"]),)
        if not joint_indices:
            continue
        distance = aabb_xy_distance(robot_xy, record["aabb_center"], record["aabb_size"])
        category = record.get("category") or "Container"
        targets.append(
            InteractionTarget(
                kind="container",
                name=str(record["name"]),
                label=f"{category} {record['name']} joint={list(joint_indices)}",
                distance_m=distance,
                record=record,
                joint_indices=joint_indices,
            )
        )
    targets.sort(key=lambda row: (row.distance_m, row.kind, row.name))
    return targets


def door_state(env, target: InteractionTarget) -> dict[str, Any]:
    leaves = []
    for child_name in target.record.get("hinge_body_names", []):
        try:
            door = Door(child_name, env.current_data)
            joint_index = int(door.get_hinge_joint_index())
            joint_range = [float(value) for value in door.get_joint_range(joint_index)]
            closed_value, open_value = probe.joint_closed_open_values(joint_range)
            value = float(door.get_joint_position(joint_index))
            span = open_value - closed_value
            fraction = 0.0 if abs(span) < 1e-9 else (value - closed_value) / span
            leaves.append(
                {
                    "name": child_name,
                    "joint_name": door.joint_names[joint_index],
                    "value": value,
                    "open_fraction": float(np.clip(fraction, 0.0, 1.0)),
                }
            )
        except Exception as exc:
            leaves.append({"name": child_name, "error": str(exc)})
    fractions = [row["open_fraction"] for row in leaves if "open_fraction" in row]
    return {
        "kind": "door",
        "name": target.name,
        "state": (
            "unknown"
            if not fractions
            else "open"
            if min(fractions) >= 0.8
            else "closed"
            if max(fractions) <= 0.2
            else "partial"
        ),
        "leaves": leaves,
    }


def container_state(env, target: InteractionTarget) -> dict[str, Any]:
    joints_by_index = {
        int(joint["joint_index"]): joint for joint in target.record.get("joints", [])
    }
    rows = []
    for joint_index in target.joint_indices:
        joint = joints_by_index[joint_index]
        value = probe.joint_value_by_name(env, joint["joint_name"])
        span = float(joint["open_value"]) - float(joint["closed_value"])
        fraction = 0.0 if abs(span) < 1e-9 else (value - float(joint["closed_value"])) / span
        rows.append(
            {
                "joint_index": joint_index,
                "joint_name": joint["joint_name"],
                "value": value,
                "open_fraction": float(np.clip(fraction, 0.0, 1.0)),
            }
        )
    fractions = [row["open_fraction"] for row in rows]
    return {
        "kind": "container",
        "name": target.name,
        "state": (
            "open"
            if fractions and min(fractions) >= 0.8
            else "closed"
            if fractions and max(fractions) <= 0.2
            else "partial"
        ),
        "joints": rows,
    }


def target_state(env, target: InteractionTarget) -> dict[str, Any]:
    return door_state(env, target) if target.kind == "door" else container_state(env, target)


def set_target_state(
    env,
    doorway_analysis: dict[str, Any],
    target: InteractionTarget,
    state: str,
) -> dict[str, Any]:
    if target.kind == "door":
        transition = emi.set_door_root_state(env, doorway_analysis, target.name, state)
    else:
        joints_by_index = {
            int(joint["joint_index"]): joint for joint in target.record.get("joints", [])
        }
        transitions = []
        for joint_index in target.joint_indices:
            joint = joints_by_index[joint_index]
            requested = float(joint[f"{state}_value"])
            probe.set_articulation_state_by_record(env, target.record, joint_index, requested)
            actual = probe.joint_value_by_name(env, joint["joint_name"])
            transitions.append(
                {
                    "joint_index": joint_index,
                    "joint_name": joint["joint_name"],
                    "requested_value": requested,
                    "actual_value": actual,
                }
            )
        transition = {"object_name": target.name, "state": state, "transitions": transitions}
    return {"transition": transition, "readback": target_state(env, target)}


def set_robot_pose(env, pose_7d: Any) -> None:
    robot = env.current_robot
    robot.update_control({})
    robot.robot_view.base.pose = pos_quat_to_pose_mat(np.asarray(pose_7d, dtype=float))
    try:
        robot.robot_view.base.joint_vel = np.zeros_like(robot.robot_view.base.joint_vel)
    except (AttributeError, ValueError):
        pass
    mujoco.mj_forward(env.current_model, env.current_data)
    env.camera_manager.registry.update_all_cameras(env)


def initialize_scene_state(
    ctx,
    candidate: dict[str, Any],
    state_preset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    container_builder.open_all_available_doors(ctx)
    doorway_analysis = emi.collect_runtime_doorway_analysis(ctx.env)
    _, containers = probe.collect_scene_records(ctx)
    container_builder.close_all_containers(ctx.env, containers)
    if state_preset == "mixed_test":
        emi.set_door_root_state(
            ctx.env,
            doorway_analysis,
            candidate["selected_mixed_door_root"],
            "closed",
        )
    elif state_preset == "visualization":
        target_container = next(
            row for row in containers if row["name"] == candidate["container_name"]
        )
        joints_by_index = {
            int(joint["joint_index"]): joint for joint in target_container["joints"]
        }
        for joint_index in candidate["joint_sequence"]:
            joint = joints_by_index[int(joint_index)]
            probe.set_articulation_state_by_record(
                ctx.env, target_container, int(joint_index), float(joint["open_value"])
            )
    else:
        raise ValueError(f"Unsupported state preset: {state_preset}")
    set_robot_pose(ctx.env, candidate["source_robot_base_pose"])
    door_records = emi.collect_interactive_door_root_object_records(ctx.env, doorway_analysis)
    return doorway_analysis, door_records, containers


def physics_step(ctx, action: dict[str, Any]) -> None:
    sim_dt_ms = round(float(ctx.env.current_model.opt.timestep) * 1000.0)
    ctrl_dt_ms = float(ctx.cfg.ctrl_dt_ms)
    policy_dt_ms = float(ctx.cfg.policy_dt_ms)
    if sim_dt_ms <= 0 or ctrl_dt_ms % sim_dt_ms != 0:
        raise ValueError(f"Invalid control/simulation dt: ctrl={ctrl_dt_ms}ms sim={sim_dt_ms}ms")
    if policy_dt_ms % ctrl_dt_ms != 0:
        raise ValueError(f"Invalid policy/control dt: policy={policy_dt_ms}ms ctrl={ctrl_dt_ms}ms")
    n_sim_steps = int(ctrl_dt_ms // sim_dt_ms)
    n_ctrl_steps = int(policy_dt_ms // ctrl_dt_ms)
    action = {key: value for key, value in action.items() if key != "done"}
    ctx.env.current_robot.update_control(action)
    for _ in range(n_ctrl_steps):
        ctx.env.current_robot.compute_control()
        ctx.env.step(n_sim_steps)


def action_requires_physics(action: dict[str, Any], *, epsilon: float = 1e-7) -> bool:
    """Return whether the manual loop has an active base command.

    An idle inspection loop should not keep integrating a dynamically settling
    robot.  Holding the last controller target without advancing MuJoCo keeps
    the robot visually stable until the user presses a movement key.
    """
    base = action.get("base")
    return base is not None and float(np.linalg.norm(np.asarray(base, dtype=float))) > epsilon


def hold_robot_static(env) -> None:
    """Stop residual robot motion without advancing simulation time."""
    robot = env.current_robot
    robot.update_control({})
    robot_view = robot.robot_view
    for group_name in robot_view.move_group_ids():
        group = robot_view.get_move_group(group_name)
        group.joint_vel = np.zeros_like(group.joint_vel)
    env.current_data.qacc[:] = 0.0
    mujoco.mj_forward(env.current_model, env.current_data)
    env.camera_manager.registry.update_all_cameras(env)


def setpos_base_step(
    env,
    base_delta: Any,
    *,
    collision_check: bool,
) -> bool:
    """Apply one planar base delta by writing the robot base pose directly."""
    delta = np.asarray(base_delta, dtype=float)
    if delta.shape != (3,):
        raise ValueError(f"base_delta must have shape (3,), got {delta.shape}")
    robot = env.current_robot
    robot_view = robot.robot_view
    current = np.asarray(robot_view.base.pose, dtype=float)
    candidate = current.copy()
    candidate[:2, 3] += delta[:2]
    yaw = math.atan2(current[1, 0], current[0, 0]) + float(delta[2])
    candidate[:2, :2] = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=float,
    )
    if collision_check and env.check_if_robot_collision_at_base_pose(
        robot_view, candidate
    ):
        hold_robot_static(env)
        return False
    robot.update_control({})
    robot_view.base.pose = candidate
    hold_robot_static(env)
    return True


def camera_pose_record(
    env,
    camera: ManualExocentricCameraController,
    *,
    step: int,
) -> dict[str, Any]:
    camera_pose = camera.pose()
    robot_pose = np.asarray(env.current_robot.robot_view.base.pose, dtype=float)
    robot_rotation = robot_pose[:3, :3]
    robot_position = robot_pose[:3, 3]
    relative_position = robot_rotation.T @ (camera_pose.position - robot_position)
    relative_forward = robot_rotation.T @ camera_pose.forward
    robot_yaw = math.atan2(robot_pose[1, 0], robot_pose[0, 0])
    relative_yaw = ManualInteractiveNavPolicy.wrap_to_pi(camera_pose.yaw - robot_yaw)
    return {
        "schema_version": "manual_camera_pose_log_v1",
        "timestamp": time.time(),
        "step": int(step),
        "camera_name": camera.camera_name,
        "world_position_m": camera_pose.position.astype(float).tolist(),
        "world_forward": camera_pose.forward.astype(float).tolist(),
        "robot_position_m": robot_position.astype(float).tolist(),
        "robot_yaw_deg": math.degrees(robot_yaw),
        "relative_position_robot_m": relative_position.astype(float).tolist(),
        "relative_forward_robot": relative_forward.astype(float).tolist(),
        "relative_yaw_deg": math.degrees(relative_yaw),
        "pitch_deg": math.degrees(camera_pose.pitch),
        "fov_deg": float(camera.fov_deg),
    }


def robot_pose_text(env) -> str:
    pose = np.asarray(env.current_robot.robot_view.base.pose, dtype=float)
    yaw = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
    return f"robot x={pose[0, 3]:.2f} y={pose[1, 3]:.2f} yaw={yaw:.1f}deg"


def normalize_cv2_key(key_code: int) -> str | None:
    if key_code < 0:
        return None
    code = int(key_code) & 0xFF
    if code == 27:
        return "esc"
    if code == 32:
        return "space"
    if 0 <= code < 128:
        char = chr(code).lower()
        return char if len(char) == 1 else None
    return None


def compose_visualization(
    env,
    camera: ManualExocentricCameraController,
    nearest: InteractionTarget | None,
    paused: bool,
    last_message: str,
    active_keys: set[str],
    base_control_mode: str,
) -> np.ndarray:
    import cv2

    frames = [("external", env.render_rgb_frame(camera.camera_name))]
    if "head_camera" in env.camera_manager.registry:
        frames.append(("head", env.render_rgb_frame("head_camera")))
    normalized = []
    target_height = min(int(np.asarray(frame).shape[0]) for _, frame in frames)
    for label, frame in frames:
        rgb = np.asarray(frame)
        scale = target_height / rgb.shape[0]
        resized = cv2.resize(rgb, (int(round(rgb.shape[1] * scale)), target_height))
        cv2.putText(
            resized,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        normalized.append(resized)
    panel = np.concatenate(normalized, axis=1)
    status_height = 150
    canvas = np.zeros((panel.shape[0] + status_height, panel.shape[1], 3), dtype=np.uint8)
    canvas[status_height:] = panel
    camera_log = camera_pose_record(env, camera, step=-1)
    nearest_text = "nearest: none"
    if nearest is not None:
        nearest_text = f"nearest: {nearest.kind} {nearest.name} d={nearest.distance_m:.2f}m"
    lines = [
        (
            f"house=783 preset=manual mixed | {'PAUSED' if paused else 'RUNNING'} "
            f"| base={base_control_mode} | keys={','.join(sorted(active_keys)) or '-'}"
        ),
        robot_pose_text(env),
        nearest_text,
        (
            f"camera rel={np.round(camera_log['relative_position_robot_m'], 2).tolist()} "
            f"yaw_rel={camera_log['relative_yaw_deg']:.1f} "
            f"pitch={camera_log['pitch_deg']:.1f}"
        ),
        "W/S move  A/D turn  O/P open/close nearest  I/K/J/L camera move  ;/' yaw  ./ pitch",
        f"R reset robot  C reset camera  V capture  Space pause  Esc quit | {last_message}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line[:180],
            (12, 22 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def capture_frame(env, camera_name: str, output_dir: Path, step: int) -> Path:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"house_783_manual_step_{step:06d}_{timestamp}.png"
    Image.fromarray(np.asarray(env.render_rgb_frame(camera_name))).save(path)
    return path


def process_event(
    event: ManualControlEvent,
    *,
    ctx,
    candidate: dict[str, Any],
    doorway_analysis: dict[str, Any],
    door_records: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    camera: ManualExocentricCameraController,
    interaction_distance_m: float,
    capture_dir: Path,
    step: int,
    paused: bool,
) -> tuple[bool, bool, str]:
    if event.name == "quit":
        return True, paused, "quit requested"
    if event.name == "toggle_pause":
        return False, not paused, "pause toggled"
    if event.name == "reset_robot":
        set_robot_pose(ctx.env, candidate["source_robot_base_pose"])
        return False, paused, "robot reset"
    if event.name == "reset_camera":
        camera.reset()
        camera.update_registered_camera(ctx.env)
        return False, paused, "camera reset"
    if event.name == "capture_frame":
        path = capture_frame(ctx.env, camera.camera_name, capture_dir, step)
        return False, paused, f"captured {path.name}"
    if event.name not in {"open_nearest", "close_nearest"}:
        return False, paused, event.name
    targets = interaction_targets(
        ctx.env, candidate, doorway_analysis, door_records, containers
    )
    if not targets:
        return False, paused, "no interactive objects"
    nearest = targets[0]
    if nearest.distance_m > interaction_distance_m:
        return (
            False,
            paused,
            f"nearest {nearest.kind} is {nearest.distance_m:.2f}m away (limit {interaction_distance_m:.2f}m)",
        )
    state = "open" if event.name == "open_nearest" else "closed"
    result = set_target_state(ctx.env, doorway_analysis, nearest, state)
    message = f"{state} {nearest.kind} {nearest.name}: {result['readback']['state']}"
    print(json.dumps(result, ensure_ascii=False, default=str))
    return False, paused, message


def run(args: argparse.Namespace) -> int:
    candidate, payload = load_candidate(args.catalog, args.case_id)
    ctx, episode = load_context(candidate, payload, args.capture_dir)
    policy = None
    camera_log_handle = None
    visualize = bool(args.visualize)
    try:
        doorway_analysis, door_records, containers = initialize_scene_state(
            ctx, candidate, args.state_preset
        )
        if args.camera_mode == "robot_over_shoulder":
            camera = ManualExocentricCameraController.from_robot_pose(
                camera_name="manual_exo_camera",
                robot_pose=np.asarray(ctx.env.current_robot.robot_view.base.pose, dtype=float),
                position_offset_robot=np.asarray(
                    args.camera_position_offset_robot, dtype=float
                ),
                lookat_offset_robot=np.asarray(args.camera_lookat_offset_robot, dtype=float),
                fov_deg=args.camera_fov,
            )
        else:
            camera = ManualExocentricCameraController.from_spherical(
                camera_name="manual_exo_camera",
                target=np.asarray(args.camera_target, dtype=float),
                distance=args.camera_distance,
                azimuth_deg=args.camera_azimuth,
                elevation_deg=args.camera_elevation,
                fov_deg=args.camera_fov,
            )
        camera.register(ctx.env)
        camera_log_path = args.camera_log_path
        if camera_log_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            camera_log_path = args.capture_dir / f"house_783_camera_pose_{timestamp}.jsonl"
        camera_log_path.parent.mkdir(parents=True, exist_ok=True)
        camera_log_handle = camera_log_path.open("w", buffering=1)
        print(f"Camera pose log: {camera_log_path}")
        if args.capture_on_start:
            initial_capture = capture_frame(
                ctx.env, camera.camera_name, args.capture_dir, step=0
            )
            print(f"Captured initial external-camera frame: {initial_capture}")
        policy = ManualInteractiveNavPolicy(
            config=ctx.cfg,
            env=ctx.env,
            linear_step_m=args.linear_step,
            angular_step_rad=math.radians(args.angular_step_deg),
            camera_translation_step_m=args.camera_translation_step,
            camera_rotation_step_rad=math.radians(args.camera_rotation_step_deg),
            start_listener=args.keyboard,
        )
        initial_targets = interaction_targets(
            ctx.env, candidate, doorway_analysis, door_records, containers
        )
        selected_door_target = next(
            row
            for row in initial_targets
            if row.kind == "door" and row.name == candidate["selected_mixed_door_root"]
        )
        target_container_target = next(
            row
            for row in initial_targets
            if row.kind == "container" and row.name == candidate["container_name"]
        )
        if args.keyboard:
            print("Keyboard listener active. Focus may remain on the visualization window.")
        print(
            json.dumps(
                {
                    "case_id": candidate["case_id"],
                    "house_index": candidate["house_index"],
                    "source_episode_index": candidate["source_episode_index"],
                    "scene_dataset": episode["scene_dataset"],
                    "state_preset": args.state_preset,
                    "selected_door": candidate["selected_mixed_door_root"],
                    "target_container": candidate["container_name"],
                    "target_object": candidate["object_name"],
                    "camera_mode": args.camera_mode,
                    "base_control_mode": args.base_control_mode,
                    "initial_camera_pose": camera_pose_record(ctx.env, camera, step=0),
                    "camera_log_path": str(camera_log_path),
                    "initial_selected_door_state": target_state(
                        ctx.env, selected_door_target
                    ),
                    "initial_target_container_state": target_state(
                        ctx.env, target_container_target
                    ),
                    "camera_names": list(ctx.env.camera_manager.registry.keys()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        paused = False
        should_quit = False
        last_message = "ready"
        step = 0
        last_camera_print_time = -math.inf
        window_key_deadlines: dict[str, float] = {}
        if visualize:
            import cv2

            cv2.namedWindow("Manual InteractiveNav", cv2.WINDOW_NORMAL)
        while not should_quit and (args.max_steps <= 0 or step < args.max_steps):
            started = time.perf_counter()
            now = time.monotonic()
            window_key_deadlines = {
                key: deadline
                for key, deadline in window_key_deadlines.items()
                if deadline >= now
            }
            window_pressed = set(window_key_deadlines)
            active_keys = policy.combined_pressed_keys(window_pressed)
            robot_pose = np.asarray(ctx.env.current_robot.robot_view.base.pose, dtype=float)
            camera.apply(
                policy.get_camera_command(extra_pressed=window_pressed),
                robot_pose=robot_pose,
            )
            camera.update_registered_camera(ctx.env)
            for event in policy.drain_events():
                should_quit, paused, last_message = process_event(
                    event,
                    ctx=ctx,
                    candidate=candidate,
                    doorway_analysis=doorway_analysis,
                    door_records=door_records,
                    containers=containers,
                    camera=camera,
                    interaction_distance_m=args.interaction_distance,
                    capture_dir=args.capture_dir,
                    step=step,
                    paused=paused,
                )
                if should_quit:
                    break
            if not should_quit and not paused:
                action = policy.get_action(extra_pressed=window_pressed)
                if action_requires_physics(action):
                    if args.base_control_mode == "setpos":
                        moved = setpos_base_step(
                            ctx.env,
                            action["base"],
                            collision_check=args.setpos_collision_check,
                        )
                        if not moved:
                            last_message = "setpos blocked by collision"
                    else:
                        physics_step(ctx, action)
                else:
                    hold_robot_static(ctx.env)
            camera.follow_robot_pose(
                np.asarray(ctx.env.current_robot.robot_view.base.pose, dtype=float)
            )
            camera.update_registered_camera(ctx.env)
            camera_log = camera_pose_record(ctx.env, camera, step=step)
            camera_log_handle.write(json.dumps(camera_log, ensure_ascii=False) + "\n")
            if started - last_camera_print_time >= args.camera_log_interval:
                print(
                    "CAMERA_REL "
                    + json.dumps(
                        {
                            "step": camera_log["step"],
                            "position_robot_m": camera_log[
                                "relative_position_robot_m"
                            ],
                            "yaw_deg": camera_log["relative_yaw_deg"],
                            "pitch_deg": camera_log["pitch_deg"],
                            "fov_deg": camera_log["fov_deg"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_camera_print_time = started
            targets = interaction_targets(
                ctx.env, candidate, doorway_analysis, door_records, containers
            )
            nearest = targets[0] if targets else None
            if visualize:
                frame = compose_visualization(
                    ctx.env,
                    camera,
                    nearest,
                    paused,
                    last_message,
                    active_keys,
                    args.base_control_mode,
                )
                cv2.imshow("Manual InteractiveNav", frame)
                window_key = normalize_cv2_key(cv2.waitKeyEx(1))
                if window_key == "esc":
                    should_quit = True
                elif window_key in WINDOW_CONTINUOUS_KEYS:
                    window_key_deadlines[window_key] = (
                        time.monotonic() + args.window_key_hold_sec
                    )
                elif window_key in ManualInteractiveNavPolicy.EDGE_EVENTS:
                    if window_key not in policy.pressed_keys():
                        policy.press_key(window_key)
                        policy.release_key(window_key)
            step += 1
            if args.fps > 0:
                delay = 1.0 / args.fps - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
        print(f"Manual InteractiveNav finished after {step} loop steps")
        return 0
    finally:
        if camera_log_handle is not None:
            camera_log_handle.close()
        if policy is not None:
            policy.close()
        if visualize:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        probe.close_context(ctx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual RBY1 navigation and oracle articulation inspection for a mixed scene."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--case_id", default=DEFAULT_CASE_ID)
    parser.add_argument("--state_preset", choices=("mixed_test", "visualization"), default="mixed_test")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_steps", type=int, default=0, help="0 runs until Esc")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--window_key_hold_sec",
        type=float,
        default=0.30,
        help="OpenCV fallback duration for one movement-key event",
    )
    parser.add_argument("--interaction_distance", type=float, default=1.75)
    parser.add_argument(
        "--base_control_mode",
        choices=("setpos", "relative_action"),
        default="setpos",
    )
    parser.add_argument(
        "--setpos_collision_check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--linear_step", type=float, default=0.035)
    parser.add_argument("--angular_step_deg", type=float, default=2.5)
    parser.add_argument("--camera_translation_step", type=float, default=0.12)
    parser.add_argument("--camera_rotation_step_deg", type=float, default=2.0)
    parser.add_argument(
        "--camera_mode",
        choices=("robot_over_shoulder", "spherical"),
        default="robot_over_shoulder",
    )
    parser.add_argument(
        "--camera_position_offset_robot",
        type=float,
        nargs=3,
        default=DEFAULT_OVER_SHOULDER_POSITION.tolist(),
        metavar=("BACK", "LEFT", "UP"),
    )
    parser.add_argument(
        "--camera_lookat_offset_robot",
        type=float,
        nargs=3,
        default=DEFAULT_OVER_SHOULDER_LOOKAT.tolist(),
        metavar=("FORWARD", "LEFT", "UP"),
    )
    parser.add_argument("--camera_target", type=float, nargs=3, default=DEFAULT_CAMERA_TARGET.tolist())
    parser.add_argument("--camera_distance", type=float, default=24.0)
    parser.add_argument("--camera_azimuth", type=float, default=345.0)
    parser.add_argument("--camera_elevation", type=float, default=-65.0)
    parser.add_argument("--camera_fov", type=float, default=65.0)
    parser.add_argument("--camera_log_interval", type=float, default=0.5)
    parser.add_argument("--camera_log_path", type=Path, default=None)
    parser.add_argument("--capture_dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--capture_on_start", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
