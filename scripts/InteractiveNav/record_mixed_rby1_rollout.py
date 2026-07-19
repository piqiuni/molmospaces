"""Record a mixed V3 door -> fridge rollout with the real RBY1 policies.

The runner keeps the V3 scene/object state as the source of truth, uses the
existing RBY1 door/container policy executor for the two manipulation phases,
and hands semantic robot/articulation state between the task-specific runners.
It intentionally does not fall back to MuJoCo force-drive when a policy fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.utils.pose import pose_mat_to_7d, pos_quat_to_pose_mat
from scripts.InteractiveNav import capture_mixed_gt_storyboard as storyboard
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import visualize_mixed_interaction_benchmark as mixed_viz


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts/InteractiveNav/output/mixed_rby1_rollout"
CAMERAS = ("head_camera", "camera_follower")
SCHEMA_VERSION = "mixed_rby1_rollout_v1"


@dataclass(frozen=True)
class SegmentPlan:
    name: str
    interaction_kind: str
    target_name: str
    joint_index: int
    start_pose: list[float]
    operation_pose: list[float]
    base_path_length_m: float
    articulation_overrides: dict[str, float]


class FrameCollector:
    """Collect camera streams from multiple task-specific RBY1 segments."""

    def __init__(self) -> None:
        self.frames: dict[str, list[np.ndarray]] = {name: [] for name in CAMERAS}
        self.events: list[dict[str, Any]] = []

    def callback(self, segment_name: str):
        def _callback(camera_name: str, frame: np.ndarray, metadata: dict[str, Any]) -> None:
            if camera_name not in self.frames:
                self.frames[camera_name] = []
            self.frames[camera_name].append(np.asarray(frame, dtype=np.uint8).copy())
            self.events.append(
                {
                    "segment": segment_name,
                    "camera": camera_name,
                    **probe.to_jsonable(metadata),
                    "frame_index": len(self.frames[camera_name]) - 1,
                }
            )

        return _callback


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )
    path.chmod(0o644)


def load_episode(path: Path, episode_index: int | None, case_id: str | None):
    episodes = storyboard.load_episodes(path)
    index, episode, selection = storyboard.choose_episode(
        episodes,
        episode_index=episode_index,
        case_id=case_id,
    )
    return index, episode, selection


def request_args(
    *,
    house_index: int,
    interaction_kind: str,
    target_name: str,
    joint_index: int,
    args: argparse.Namespace,
) -> argparse.Namespace:
    request = probe.RBY1InteractionRequest(
        house_ind=house_index,
        interaction_kind=interaction_kind,
        target_name=target_name,
        joint_index=joint_index,
        robot_pose_mode="auto",
        door_arm=args.door_arm,
        container_arm=args.container_arm,
        approach_distance=args.approach_distance,
        min_base_clearance=args.min_base_clearance,
        max_approach_distance=args.max_approach_distance,
        max_base_adjustment_distance=args.max_base_adjustment_distance,
        max_base_adjustment_steps=args.max_base_adjustment_steps,
        door_max_steps_per_waypoint=args.door_max_steps_per_waypoint,
        door_max_planning_reattempts=args.door_max_planning_reattempts,
        door_joint_position_tolerance=args.door_joint_position_tolerance,
        door_articulation_delta_deg=args.door_articulation_delta_deg,
        allow_force_fallback=args.allow_force_fallback,
        force_fallback_target_fraction=args.force_fallback_target_fraction,
        force_fallback_max_steps=args.force_fallback_max_steps,
        success_threshold=args.success_threshold,
        max_steps=args.max_steps,
        container_max_steps_per_waypoint=args.container_max_steps_per_waypoint,
        container_max_batch_plan_attempts=args.container_max_batch_plan_attempts,
        container_max_planning_reattempts=args.container_max_planning_reattempts,
        scene_dataset="procthor-10k",
        data_split=args.data_split,
        variant=args.variant,
        seed=args.seed,
    )
    return probe._rby1_request_to_args(request)


def prepare_operation_spec(
    episode: dict[str, Any],
    *,
    house_index: int,
    interaction_kind: str,
    target_name: str,
    joint_index: int,
    start_pose: list[float],
    args: argparse.Namespace,
) -> tuple[EpisodeSpec, dict[str, Any], list[float]]:
    """Build a policy episode, replacing only its semantic initial state/pose."""
    operation_args = request_args(
        house_index=house_index,
        interaction_kind=interaction_kind,
        target_name=target_name,
        joint_index=joint_index,
        args=args,
    )
    episode_spec, target_meta = probe.prepare_rby1_interaction_episode(operation_args)
    payload = episode_spec.model_dump(mode="json")
    operation_pose = list(payload["task"]["robot_base_pose"])
    payload["task"]["robot_base_pose"] = list(start_pose)
    payload["scene_modifications"] = copy.deepcopy(episode.get("scene_modifications", {}))

    target_name_from_episode = episode.get("interactive_nav", {}).get("target", {}).get(
        "selected_instance"
    )
    if interaction_kind == "container" and target_name_from_episode:
        target_pose = payload["scene_modifications"].get("object_poses", {}).get(
            target_name_from_episode
        )
        if target_pose is not None:
            payload["task"]["pickup_obj_start_pose"] = list(target_pose)
    return EpisodeSpec.model_validate(payload), target_meta, operation_pose


def pose_for_xy(xy: np.ndarray, next_xy: np.ndarray) -> np.ndarray:
    direction = np.asarray(next_xy, dtype=float) - np.asarray(xy, dtype=float)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, 0.0], dtype=float)
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[:2, 3] = np.asarray(xy, dtype=float)
    return pose


def compress_path(path: np.ndarray, spacing_m: float = 0.12) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path
    selected = [path[0]]
    accumulated = 0.0
    previous = path[0]
    for point in path[1:-1]:
        accumulated += float(np.linalg.norm(point - previous))
        if accumulated >= spacing_m:
            selected.append(point)
            accumulated = 0.0
        previous = point
    selected.append(path[-1])
    return np.asarray(selected, dtype=float)


def compute_navigation_path(
    episode: dict[str, Any],
    *,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    door_state: str,
    required_door_root: str,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], float]:
    """Compute a live occupancy-map path and turn it into base pose waypoints."""
    scene_args = argparse.Namespace(
        scene_dataset="procthor-10k",
        data_split=args.data_split,
        robot="rby1",
        variant=args.variant,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    ctx = probe.load_scene_context(scene_args, int(episode["house_index"]))
    try:
        probe.apply_episode_scene_state(ctx.env, episode)
        live_map, doorway_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        if door_state == "open":
            emi.set_door_root_state(ctx.env, doorway_analysis, required_door_root, "open")
            live_map = emi.build_live_procthor_map(
                ctx.env.current_model,
                ctx.env.current_data,
                model_path=str(ctx.env.current_model_path),
                px_per_m=args.px_per_m,
                agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
                open_threshold=args.open_threshold,
                treat_all_non_interactive_doorways_as_open=True,
            )
        path = emi.compute_path_from_map(
            live_map,
            np.asarray(start_xy, dtype=float),
            np.asarray(goal_xy, dtype=float),
            downscale_factor=1,
        )
        if path is None:
            raise RuntimeError(
                f"No {door_state} navigation path from {start_xy.tolist()} to {goal_xy.tolist()}"
            )
        path = compress_path(path, spacing_m=args.nav_waypoint_spacing_m)
        poses = [
            pose_for_xy(path[index], path[min(index + 1, len(path) - 1)])
            for index in range(len(path))
        ]
        return poses, float(emi.path_length(path))
    finally:
        probe.close_context(ctx)


def target_joint_open_value(target_meta: dict[str, Any]) -> float:
    return float(max(target_meta["joint_range"]))


def semantic_fraction(result: dict[str, Any]) -> float:
    if "semantic_open_fraction" in result:
        return float(result["semantic_open_fraction"])
    closed = float(result.get("semantic_closed_value", 0.0))
    opened = float(result.get("semantic_open_value", 1.0))
    value = float(result.get("final_joint_position", closed))
    return probe.semantic_open_fraction(value, closed, opened)


def save_combined_videos(output_dir: Path, collector: FrameCollector, fps: float) -> dict[str, str]:
    output_paths = {}
    for camera_name, frames in collector.frames.items():
        if not frames:
            continue
        path = output_dir / f"mixed_rby1_{camera_name}.mp4"
        probe.save_frames_to_mp4(frames, str(path), fps=fps)
        output_paths[camera_name] = str(path)
    return output_paths


def run(args: argparse.Namespace) -> int:
    episode_index, episode, selection = load_episode(
        args.benchmark,
        args.episode_index,
        args.case_id,
    )
    annotations = mixed_viz.extract_episode_annotations(episode)
    run_dir = args.output_dir / (
        f"episode_{episode_index:04d}_h{annotations['house_index']}_"
        f"{storyboard.safe_slug(annotations['case_id'], 72)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    start_pose = list(episode["task"]["robot_base_pose"])
    door_name = annotations["channel_object_names"][0]
    door_root = annotations["required_door_roots"][0]
    container_name = annotations["container_name"]
    door_interaction = next(
        row
        for row in annotations["interactions"]
        if str(row["type"]).startswith("channel_")
    )
    container_interaction = storyboard.container_interactions(episode)[0]

    door_spec, door_meta, door_operation_pose = prepare_operation_spec(
        episode,
        house_index=annotations["house_index"],
        interaction_kind="door",
        target_name=door_name,
        joint_index=int(door_interaction["joint_index"]),
        start_pose=start_pose,
        args=args,
    )
    door_operation_pose_array = np.asarray(door_operation_pose, dtype=float)
    door_path, door_path_length = compute_navigation_path(
        episode,
        start_xy=np.asarray(start_pose[:2], dtype=float),
        goal_xy=door_operation_pose_array[:2],
        door_state="closed",
        required_door_root=door_root,
        args=args,
    )
    if args.plan_only:
        fridge_spec, fridge_meta, fridge_operation_pose = prepare_operation_spec(
            episode,
            house_index=annotations["house_index"],
            interaction_kind="container",
            target_name=container_name,
            joint_index=int(container_interaction["joint_index"]),
            start_pose=door_operation_pose,
            args=args,
        )
        del fridge_spec
        fridge_path, fridge_path_length = compute_navigation_path(
            episode,
            start_xy=np.asarray(door_operation_pose[:2], dtype=float),
            goal_xy=np.asarray(fridge_operation_pose[:2], dtype=float),
            door_state="open",
            required_door_root=door_root,
            args=args,
        )
        plan_payload = {
            "schema_version": SCHEMA_VERSION,
            "episode_index": episode_index,
            "case_id": annotations["case_id"],
            "selection": selection,
            "door_operation_pose": door_operation_pose,
            "fridge_operation_pose": fridge_operation_pose,
            "door_path_length_m": door_path_length,
            "fridge_path_length_m": fridge_path_length,
            "door_waypoint_count": len(door_path),
            "fridge_waypoint_count": len(fridge_path),
            "door_target_meta": door_meta,
            "fridge_target_meta": fridge_meta,
        }
        write_json(run_dir / "plan.json", plan_payload)
        print(json.dumps({"output_dir": str(run_dir), "plan": "plan.json"}, ensure_ascii=False))
        return 0

    collector = FrameCollector()
    door_output = run_dir / "door"
    door_result = probe.execute_rby1_whole_body_interaction(
        probe.build_rby1_interaction_config(request_args(
            house_index=annotations["house_index"],
            interaction_kind="door",
            target_name=door_name,
            joint_index=int(door_interaction["joint_index"]),
            args=args,
        )),
        door_spec,
        interaction_kind="door",
        variant=args.variant,
        output_dir=door_output,
        camera_names=CAMERAS,
        max_steps=args.max_steps,
        video_fps=args.video_fps,
        base_adjustment_path=door_path,
        max_base_adjustment_steps=args.max_base_adjustment_steps,
        initial_state_episode=episode,
        frame_callback=collector.callback("nav_to_door_and_open"),
        allow_force_fallback=args.allow_force_fallback,
        force_fallback_target_fraction=args.force_fallback_target_fraction,
        force_fallback_max_steps=args.force_fallback_max_steps,
    )
    write_json(door_output / "result.json", door_result)
    if not door_result["success"] or semantic_fraction(door_result) < args.required_open_fraction:
        raise RuntimeError(
            "RBY1 door phase did not meet the required opening threshold: "
            f"success={door_result['success']} fraction={semantic_fraction(door_result):.3f}"
        )

    final_robot_state = door_result["final_robot_state"]
    fridge_start_pose = final_robot_state["base_pose"]
    fridge_spec, fridge_meta, fridge_operation_pose = prepare_operation_spec(
        episode,
        house_index=annotations["house_index"],
        interaction_kind="container",
        target_name=container_name,
        joint_index=int(container_interaction["joint_index"]),
        start_pose=fridge_start_pose,
        args=args,
    )
    fridge_operation_pose_array = np.asarray(fridge_operation_pose, dtype=float)
    fridge_path, fridge_path_length = compute_navigation_path(
        episode,
        start_xy=np.asarray(fridge_start_pose[:2], dtype=float),
        goal_xy=fridge_operation_pose_array[:2],
        door_state="open",
        required_door_root=door_root,
        args=args,
    )
    door_joint_name = door_meta["joint_name"]
    door_open_value = target_joint_open_value(door_meta)
    fridge_output = run_dir / "fridge"
    fridge_result = probe.execute_rby1_whole_body_interaction(
        probe.build_rby1_interaction_config(request_args(
            house_index=annotations["house_index"],
            interaction_kind="container",
            target_name=container_name,
            joint_index=int(container_interaction["joint_index"]),
            args=args,
        )),
        fridge_spec,
        interaction_kind="container",
        variant=args.variant,
        output_dir=fridge_output,
        camera_names=CAMERAS,
        max_steps=args.max_steps,
        video_fps=args.video_fps,
        base_adjustment_path=fridge_path,
        max_base_adjustment_steps=args.max_base_adjustment_steps,
        initial_state_episode=episode,
        initial_articulation_overrides={door_joint_name: door_open_value},
        initial_robot_state=final_robot_state,
        frame_callback=collector.callback("nav_to_fridge_and_open"),
        hold_base_during_policy=True,
        allow_force_fallback=args.allow_force_fallback,
        force_fallback_target_fraction=args.force_fallback_target_fraction,
        force_fallback_max_steps=args.force_fallback_max_steps,
    )
    write_json(fridge_output / "result.json", fridge_result)
    if not fridge_result["success"] or semantic_fraction(fridge_result) < args.required_open_fraction:
        raise RuntimeError(
            "RBY1 fridge phase did not meet the required opening threshold: "
            f"success={fridge_result['success']} fraction={semantic_fraction(fridge_result):.3f}"
        )

    video_paths = save_combined_videos(run_dir, collector, args.video_fps)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": str(args.benchmark),
        "episode_index": episode_index,
        "case_id": annotations["case_id"],
        "house_index": annotations["house_index"],
        "selection": selection,
        "target": {
            "name": annotations["target_name"],
            "category": annotations["target_category"],
            "container_name": container_name,
        },
        "required_door": {
            "root_name": door_root,
            "leaf_name": door_name,
            "joint_name": door_meta["joint_name"],
            "joint_index": int(door_meta["joint_index"]),
        },
        "segments": [
            asdict(
                SegmentPlan(
                    name="nav_to_door_and_open",
                    interaction_kind="door",
                    target_name=door_name,
                    joint_index=int(door_meta["joint_index"]),
                    start_pose=start_pose,
                    operation_pose=door_operation_pose,
                    base_path_length_m=door_path_length,
                    articulation_overrides={},
                )
            )
            | {"result": door_result},
            asdict(
                SegmentPlan(
                    name="nav_to_fridge_and_open",
                    interaction_kind="container",
                    target_name=container_name,
                    joint_index=int(container_interaction["joint_index"]),
                    start_pose=fridge_start_pose,
                    operation_pose=fridge_operation_pose,
                    base_path_length_m=fridge_path_length,
                    articulation_overrides={door_joint_name: door_open_value},
                )
            )
            | {"result": fridge_result},
        ],
        "video_fps": args.video_fps,
        "video_paths": video_paths,
        "frame_event_count": len(collector.events),
        "frame_events": collector.events,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(run_dir), "video_paths": video_paths}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a mixed V3 episode with continuous GT navigation and RBY1 door/fridge policies."
    )
    parser.add_argument("benchmark", type=Path, nargs="?", default=DEFAULT_BENCHMARK)
    parser.add_argument("--episode_index", type=int)
    parser.add_argument("--case_id")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--data_split", choices=["train", "val"], default="val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--door_arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--container_arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--approach_distance", type=float, default=0.5)
    parser.add_argument("--min_base_clearance", type=float, default=0.15)
    parser.add_argument("--max_approach_distance", type=float, default=1.2)
    parser.add_argument("--max_base_adjustment_distance", type=float, default=0.75)
    parser.add_argument("--max_base_adjustment_steps", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--door_max_steps_per_waypoint", type=int, default=35)
    parser.add_argument("--door_max_planning_reattempts", type=int, default=8)
    parser.add_argument("--door_joint_position_tolerance", type=float, default=0.10)
    parser.add_argument("--door_articulation_delta_deg", type=float, default=7.0)
    parser.add_argument("--allow_force_fallback", action="store_true")
    parser.add_argument("--force_fallback_target_fraction", type=float, default=1.0)
    parser.add_argument("--force_fallback_max_steps", type=int, default=1500)
    parser.add_argument("--container_max_steps_per_waypoint", type=int, default=80)
    parser.add_argument("--container_max_batch_plan_attempts", type=int, default=16)
    parser.add_argument("--container_max_planning_reattempts", type=int, default=8)
    parser.add_argument("--success_threshold", type=float, default=0.67)
    parser.add_argument("--required_open_fraction", type=float, default=0.67)
    parser.add_argument("--video_fps", type=float, default=10.0)
    parser.add_argument("--px_per_m", type=float, default=100.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--nav_waypoint_spacing_m", type=float, default=0.12)
    parser.add_argument("--plan_only", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
