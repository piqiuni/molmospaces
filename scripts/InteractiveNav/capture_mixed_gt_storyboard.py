from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.utils.pose import compute_lookat_forward_up, pos_quat_to_pose_mat
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi
from scripts.InteractiveNav import interactive_nav_v3
from scripts.InteractiveNav import visualize_mixed_interaction_benchmark as mixed_viz


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("molmo_spaces.env.camera_manager").setLevel(logging.WARNING)

DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "scripts/InteractiveNav/output/mixed_gt_storyboard"
)
STORY_SCHEMA_VERSION = "mixed_gt_storyboard_v1"


@dataclass(frozen=True)
class StoryStep:
    index: int
    name: str
    title: str
    phase: str
    robot_pose: list[float]
    door_state: str
    container_state: str
    pathpoint_source: str
    pathpoint_index: int


@dataclass(frozen=True)
class ShoulderCameraConfig:
    name: str = "mixed_story_shoulder_camera"
    behind_m: float = 0.72
    right_m: float = 0.34
    height_m: float = 1.72
    lookahead_m: float = 1.75
    look_height_m: float = 1.08
    fov_deg: float = 62.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    )
    temporary.chmod(0o644)
    temporary.replace(path)


def safe_slug(value: str, max_len: int = 90) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return (slug or "mixed-story")[:max_len]


def load_episodes(path: Path) -> list[dict[str, Any]]:
    return mixed_viz.load_benchmark_episodes(path)


def representative_score(index: int, episode: dict[str, Any]) -> tuple[Any, ...] | None:
    nav = episode.get("interactive_nav", {})
    if nav.get("schema_version") != "interactive_nav_v3":
        return None
    if set(nav.get("interaction_domains", [])) != {"channel", "container"}:
        return None
    target = nav.get("target", {})
    if target.get("container_category") != "Fridge":
        return None
    validation = nav.get("generation_validation", {})
    navigation = validation.get("navigation_validation", {})
    required_roots = list(
        nav.get("initial_state", {}).get("required_door_roots_closed", [])
    )
    crossed_roots = list(navigation.get("all_open_path_crossed_door_roots", []))
    channel = [
        row
        for row in nav.get("interactions", [])
        if str(row.get("type", "")).startswith("channel_")
    ]
    containers = [
        row
        for row in nav.get("interactions", [])
        if str(row.get("type", "")).startswith("container_")
    ]
    if len(required_roots) != 1 or len(crossed_roots) != 1:
        return None
    if len(channel) != 1 or len(containers) != 1:
        return None
    if validation.get("minimal_plan_verified") is not True:
        return None
    if navigation.get("initial_state_path_found") is not False:
        return None
    if navigation.get("oracle_restored_path_found") is not True:
        return None
    if navigation.get("interaction_pose_collision_free") is not True:
        return None

    target_preference = {
        "apple": 0,
        "potato": 1,
        "tomato": 2,
        "egg": 3,
        "lettuce": 4,
    }
    target_rank = target_preference.get(str(target.get("category")), 20)
    path_length = float(navigation.get("all_open_path_length_m", 1e9))
    approach_length = float(navigation.get("approach_path_length_m", 0.0))
    return (
        target_rank,
        abs(path_length - 6.0),
        abs(approach_length - 2.5),
        index,
    )


def choose_episode(
    episodes: list[dict[str, Any]],
    *,
    episode_index: int | None,
    case_id: str | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    if episode_index is not None:
        if not 0 <= episode_index < len(episodes):
            raise ValueError(f"Episode index out of range: {episode_index}")
        selected = episodes[episode_index]
        interactive_nav_v3.validate_mixed_v3_episode(selected)
        return episode_index, selected, {"selection_mode": "explicit_episode_index"}
    if case_id is not None:
        for index, episode in enumerate(episodes):
            if episode.get("interactive_nav", {}).get("case_id") == case_id:
                interactive_nav_v3.validate_mixed_v3_episode(episode)
                return index, episode, {"selection_mode": "explicit_case_id"}
        raise ValueError(f"Mixed case_id not found: {case_id}")

    candidates = []
    for index, episode in enumerate(episodes):
        score = representative_score(index, episode)
        if score is not None:
            candidates.append((score, index, episode))
    if not candidates:
        raise ValueError("No representative one-required-door + Fridge mixed episode was found")
    candidates.sort(key=lambda row: row[0])
    score, index, episode = candidates[0]
    interactive_nav_v3.validate_mixed_v3_episode(episode)
    return index, episode, {
        "selection_mode": "automatic_representative",
        "score": list(score),
        "candidate_count": len(candidates),
        "criteria": {
            "container_category": "Fridge",
            "required_door_root_count": 1,
            "all_open_crossed_door_root_count": 1,
            "channel_interaction_count": 1,
            "container_interaction_count": 1,
            "minimal_plan_verified": True,
            "preferred_visible_target_category": "apple",
        },
    }


def yaw_from_pose(pose_7d: np.ndarray) -> float:
    w, x, y, z = np.asarray(pose_7d[3:7], dtype=float)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_facing_xy(xy: np.ndarray, target_xy: np.ndarray, z: float = 0.0) -> np.ndarray:
    direction = np.asarray(target_xy, dtype=float) - np.asarray(xy, dtype=float)
    if float(np.linalg.norm(direction)) < 1e-8:
        raise ValueError("Cannot construct a heading from coincident pathpoints")
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    return np.asarray(
        [
            float(xy[0]),
            float(xy[1]),
            float(z),
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ],
        dtype=float,
    )


def path_lookahead_xy(path: np.ndarray, start_index: int, distance_m: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        raise ValueError("A path needs at least two points to define a heading")
    start_index = min(max(int(start_index), 0), len(path) - 2)
    origin = path[start_index]
    accumulated = 0.0
    for index in range(start_index + 1, len(path)):
        accumulated += float(np.linalg.norm(path[index] - path[index - 1]))
        if accumulated >= distance_m:
            return path[index]
    return path[-1]


def container_interactions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in episode["interactive_nav"]["interactions"]
        if str(row["type"]).startswith("container_")
    ]


def container_joint_record(
    container: dict[str, Any], interaction: dict[str, Any]
) -> dict[str, Any]:
    joint_index = int(interaction["joint_index"])
    for joint in container["joints"]:
        if int(joint["joint_index"]) == joint_index:
            if joint["joint_name"] != interaction["joint_name"]:
                raise ValueError("Container interaction joint name disagrees with live record")
            return joint
    raise ValueError(
        f"Container interaction joint index {joint_index} was not found: {container['name']}"
    )


def set_container_state(
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    interactions: list[dict[str, Any]],
    state: str,
) -> list[dict[str, Any]]:
    if state not in {"closed", "open"}:
        raise ValueError(f"Unsupported container state: {state}")
    rows = []
    for interaction in interactions:
        joint = container_joint_record(container, interaction)
        value = float(joint[f"{state}_value"])
        probe.set_articulation_state_by_record(
            ctx.env,
            container,
            int(joint["joint_index"]),
            value,
        )
        rows.append(
            {
                "interaction_id": interaction["interaction_id"],
                "joint_index": int(joint["joint_index"]),
                "joint_name": joint["joint_name"],
                "requested_state": state,
                "requested_value": value,
                "actual_value": probe.joint_value_by_name(ctx.env, joint["joint_name"]),
            }
        )
    return rows


def shoulder_camera_pose(
    robot_pose_7d: np.ndarray,
    config: ShoulderCameraConfig,
) -> dict[str, np.ndarray]:
    robot_pose_7d = np.asarray(robot_pose_7d, dtype=float)
    yaw = yaw_from_pose(robot_pose_7d)
    forward_xy = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=float)
    right_xy = np.asarray([math.sin(yaw), -math.cos(yaw)], dtype=float)
    base = robot_pose_7d[:3]
    position = np.asarray(
        [
            base[0] - config.behind_m * forward_xy[0] + config.right_m * right_xy[0],
            base[1] - config.behind_m * forward_xy[1] + config.right_m * right_xy[1],
            base[2] + config.height_m,
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            base[0] + config.lookahead_m * forward_xy[0],
            base[1] + config.lookahead_m * forward_xy[1],
            base[2] + config.look_height_m,
        ],
        dtype=np.float32,
    )
    camera_forward, camera_up = compute_lookat_forward_up(position, target)
    return {
        "position": position,
        "target": target,
        "forward": np.asarray(camera_forward, dtype=np.float32),
        "up": np.asarray(camera_up, dtype=np.float32),
    }


def place_robot(ctx: probe.LoadedContext, robot_pose_7d: np.ndarray) -> None:
    ctx.env.current_robot.robot_view.base.pose = pos_quat_to_pose_mat(robot_pose_7d)
    mujoco.mj_forward(ctx.env.current_model, ctx.env.current_data)
    ctx.env.camera_manager.registry.update_all_cameras(ctx.env)


def register_shoulder_camera(
    ctx: probe.LoadedContext,
    robot_pose_7d: np.ndarray,
    config: ShoulderCameraConfig,
) -> dict[str, np.ndarray]:
    pose = shoulder_camera_pose(robot_pose_7d, config)
    ctx.env.camera_manager.add_camera(
        config.name,
        pose["position"],
        pose["forward"],
        pose["up"],
        fov=config.fov_deg,
    )
    return pose


def interaction_readback(episode: dict[str, Any], ctx: probe.LoadedContext) -> list[dict[str, Any]]:
    rows = []
    for interaction in episode["interactive_nav"]["interactions"]:
        rows.append(
            {
                "interaction_id": interaction["interaction_id"],
                "type": interaction["type"],
                "object_name": interaction["object_name"],
                "joint_name": interaction["joint_name"],
                "joint_index": int(interaction["joint_index"]),
                "actual_joint_value": probe.joint_value_by_name(
                    ctx.env, interaction["joint_name"]
                ),
            }
        )
    return rows


def apply_story_step_state(
    ctx: probe.LoadedContext,
    episode: dict[str, Any],
    *,
    doorway_analysis: dict[str, Any],
    required_door_root: str,
    container: dict[str, Any],
    interactions: list[dict[str, Any]],
    step: StoryStep,
) -> dict[str, Any]:
    initial_application = mixed_viz.apply_episode_initial_state(ctx, episode)
    door_transition = emi.set_door_root_state(
        ctx.env,
        doorway_analysis,
        required_door_root,
        step.door_state,
    )
    container_transitions = set_container_state(
        ctx,
        container,
        interactions,
        step.container_state,
    )
    return {
        "initial_state_application": initial_application,
        "door_transition": door_transition,
        "container_transitions": container_transitions,
    }


def build_story_steps(
    episode: dict[str, Any],
    *,
    approach_path: np.ndarray,
    door_center_xy: np.ndarray,
) -> list[StoryStep]:
    nav = episode["interactive_nav"]["generation_validation"]["navigation_validation"]
    start_xy = np.asarray(episode["task"]["robot_base_pose"][:2], dtype=float)
    start_heading_target = path_lookahead_xy(approach_path, 0, distance_m=0.65)
    start_pose = pose_facing_xy(start_xy, start_heading_target)
    approach_xy = np.asarray(nav["door_approach"]["approach_xy"], dtype=float)
    door_pose = pose_facing_xy(approach_xy, door_center_xy)
    interaction_pose = np.asarray(nav["interaction_pose"], dtype=float)
    approach_index = len(approach_path) - 1
    return [
        StoryStep(
            index=1,
            name="start",
            title="Start · door closed · fridge closed",
            phase="navigation_start",
            robot_pose=start_pose.tolist(),
            door_state="closed",
            container_state="closed",
            pathpoint_source="initial_approach_path",
            pathpoint_index=0,
        ),
        StoryStep(
            index=2,
            name="door_front_closed",
            title="Door front · door closed",
            phase="before_channel_interaction",
            robot_pose=door_pose.tolist(),
            door_state="closed",
            container_state="closed",
            pathpoint_source="initial_approach_path",
            pathpoint_index=approach_index,
        ),
        StoryStep(
            index=3,
            name="door_front_open",
            title="Door front · door open",
            phase="after_channel_interaction",
            robot_pose=door_pose.tolist(),
            door_state="open",
            container_state="closed",
            pathpoint_source="initial_approach_path",
            pathpoint_index=approach_index,
        ),
        StoryStep(
            index=4,
            name="fridge_front_closed",
            title="Fridge front · fridge closed",
            phase="before_container_interaction",
            robot_pose=interaction_pose.tolist(),
            door_state="open",
            container_state="closed",
            pathpoint_source="oracle_restored_path",
            pathpoint_index=-1,
        ),
        StoryStep(
            index=5,
            name="fridge_front_open",
            title="Fridge front · fridge open",
            phase="after_container_interaction",
            robot_pose=interaction_pose.tolist(),
            door_state="open",
            container_state="open",
            pathpoint_source="oracle_restored_path",
            pathpoint_index=-1,
        ),
    ]


def save_storyboard(output_path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    columns = 3
    grid_rows = 2
    fig, axes = plt.subplots(grid_rows, columns, figsize=(18, 8.5))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes_array, rows):
        frame = plt.imread(output_path.parent / row["image"])
        ax.imshow(frame)
        ax.set_title(f"Step {row['step']['index']}: {row['step']['title']}", fontsize=11)
        ax.axis("off")
    for ax in axes_array[len(rows) :]:
        ax.axis("off")
    fig.suptitle("Mixed GT five-step storyboard · right-rear shoulder camera", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


def draw_box(ax, scene_map, record: dict[str, Any], color: str, label: str) -> None:
    from matplotlib.patches import Rectangle

    box = emi.object_box_to_px(scene_map, record)
    if box is None:
        return
    col, row, width, height = box
    ax.add_patch(
        Rectangle(
            (col, row),
            max(width, 2.0),
            max(height, 2.0),
            facecolor="none",
            edgecolor=color,
            linewidth=3.0,
            zorder=8,
        )
    )
    ax.text(
        col + width / 2.0,
        row + height / 2.0,
        label,
        color=color,
        fontsize=8,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        zorder=9,
    )


def save_pathpoint_overview(
    output_path: Path,
    *,
    scene_map,
    approach_path: np.ndarray,
    restored_path: np.ndarray,
    steps: list[StoryStep],
    door_record: dict[str, Any],
    container: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    background = emi.make_scene_plot_background(scene_map)
    ax.imshow(background, origin="upper")
    for path, color, style, label in (
        (approach_path, "#f97316", "--", "initial approach path"),
        (restored_path, "#2563eb", "-", "oracle restored path"),
    ):
        pixels = emi.points_xy_to_px(scene_map, path)
        if pixels is not None:
            ax.plot(
                pixels[:, 1],
                pixels[:, 0],
                color=color,
                linestyle=style,
                linewidth=2.8,
                label=label,
                zorder=6,
            )
    draw_box(ax, scene_map, door_record, "#dc2626", "required door")
    draw_box(ax, scene_map, container, "#9333ea", "Fridge")
    unique_positions = [steps[0], steps[1], steps[3]]
    markers = ["o", "D", "*"]
    colors = ["#16a34a", "#f97316", "#9333ea"]
    labels = ["1 start", "2/3 door front", "4/5 fridge front"]
    for step, marker, color, label in zip(
        unique_positions, markers, colors, labels, strict=True
    ):
        xy = np.asarray(step.robot_pose[:2], dtype=float)
        pixel = emi.points_xy_to_px(scene_map, np.asarray([xy]))
        if pixel is None:
            continue
        ax.scatter(
            pixel[:, 1],
            pixel[:, 0],
            marker=marker,
            s=110,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
        ax.text(
            pixel[0, 1] + 12,
            pixel[0, 0] - 10,
            label,
            color=color,
            fontsize=10,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=11,
        )
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mixed GT storyboard pathpoints", fontsize=15)
    ax.legend(loc="lower left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor="white")
    plt.close(fig)
    output_path.chmod(0o644)


def run(args: argparse.Namespace) -> int:
    episodes = load_episodes(args.benchmark)
    episode_index, episode, selection = choose_episode(
        episodes,
        episode_index=args.episode_index,
        case_id=args.case_id,
    )
    annotations = mixed_viz.extract_episode_annotations(episode)
    run_dir = args.output_dir / (
        f"episode_{episode_index:04d}_h{annotations['house_index']}_"
        f"{safe_slug(annotations['case_id'], 68)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = run_dir

    camera_config = ShoulderCameraConfig(
        name=args.camera_name,
        behind_m=args.camera_behind_m,
        right_m=args.camera_right_m,
        height_m=args.camera_height_m,
        lookahead_m=args.camera_lookahead_m,
        look_height_m=args.camera_look_height_m,
        fov_deg=args.camera_fov_deg,
    )
    ctx = None
    try:
        ctx = container_builder.load_episode_context(args, episode)
        initial_application = mixed_viz.apply_episode_initial_state(ctx, episode)
        initial_map, doorway_analysis = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        required_door_root = annotations["required_door_roots"][0]
        door_records = emi.collect_interactive_door_root_object_records(
            ctx.env, doorway_analysis
        )
        door_record = next(
            (row for row in door_records if row["name"] == required_door_root),
            None,
        )
        if door_record is None:
            raise ValueError(f"Required door root was not found: {required_door_root}")
        _, containers = probe.collect_scene_records(ctx)
        container = next(
            (row for row in containers if row["name"] == annotations["container_name"]),
            None,
        )
        if container is None:
            raise ValueError(f"Fridge record was not found: {annotations['container_name']}")
        interactions = container_interactions(episode)
        start_xy = annotations["start_xy"]
        approach_xy = annotations["door_approach_xy"]
        interaction_xy = annotations["interaction_xy"]
        approach_path = emi.compute_path_from_map(
            initial_map, start_xy, approach_xy, downscale_factor=1
        )
        if approach_path is None:
            raise ValueError("Initial state cannot reach the validated door approach pathpoint")

        emi.set_door_root_state(
            ctx.env, doorway_analysis, required_door_root, "open"
        )
        open_map = emi.build_live_procthor_map(
            ctx.env.current_model,
            ctx.env.current_data,
            model_path=str(ctx.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=ctx.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
        )
        restored_path = emi.compute_path_from_map(
            open_map, start_xy, interaction_xy, downscale_factor=1
        )
        if restored_path is None:
            raise ValueError("Opening the required door did not restore the GT path")
        steps = build_story_steps(
            episode,
            approach_path=approach_path,
            door_center_xy=np.asarray(door_record["aabb_center"], dtype=float)[:2],
        )
        write_json(
            run_dir / "story_steps.json",
            {
                "schema_version": STORY_SCHEMA_VERSION,
                "episode_index": episode_index,
                "case_id": annotations["case_id"],
                "steps": [asdict(step) for step in steps],
            },
        )

        rows = []
        for step in steps:
            state_application = apply_story_step_state(
                ctx,
                episode,
                doorway_analysis=doorway_analysis,
                required_door_root=required_door_root,
                container=container,
                interactions=interactions,
                step=step,
            )
            robot_pose = np.asarray(step.robot_pose, dtype=float)
            place_robot(ctx, robot_pose)
            camera_pose = register_shoulder_camera(ctx, robot_pose, camera_config)
            rgb = ctx.env.render_rgb_frame(camera_config.name)
            image_path = run_dir / f"step_{step.index:02d}_{step.name}.png"
            probe.save_rgb_image(image_path, rgb)
            image_path.chmod(0o644)
            row = {
                "step": asdict(step),
                "image": image_path.name,
                "state_application": state_application,
                "interaction_readback": interaction_readback(episode, ctx),
                "camera": {
                    "config": asdict(camera_config),
                    "position": camera_pose["position"].tolist(),
                    "target": camera_pose["target"].tolist(),
                    "forward": camera_pose["forward"].tolist(),
                    "up": camera_pose["up"].tolist(),
                },
            }
            write_json(run_dir / f"step_{step.index:02d}_{step.name}.json", row)
            rows.append(row)
            print(
                f"[{step.index}/{len(steps)}] captured {step.name} "
                f"door={step.door_state} fridge={step.container_state}",
                flush=True,
            )

        storyboard_path = run_dir / "storyboard.png"
        save_storyboard(storyboard_path, rows)
        pathpoint_path = run_dir / "pathpoints_topdown.png"
        save_pathpoint_overview(
            pathpoint_path,
            scene_map=open_map,
            approach_path=approach_path,
            restored_path=restored_path,
            steps=steps,
            door_record=door_record,
            container=container,
        )
        manifest = {
            "schema_version": STORY_SCHEMA_VERSION,
            "benchmark": str(args.benchmark),
            "episode_index": episode_index,
            "case_id": annotations["case_id"],
            "house_index": annotations["house_index"],
            "selection": selection,
            "representative_scene": {
                "target_name": annotations["target_name"],
                "target_category": annotations["target_category"],
                "container_name": annotations["container_name"],
                "container_category": annotations["container_category"],
                "required_door_root": required_door_root,
                "recorded_gt_path_length_m": annotations["recorded_gt_path_length_m"],
                "recorded_approach_path_length_m": annotations[
                    "recorded_approach_path_length_m"
                ],
                "recomputed_approach_path_length_m": emi.path_length(approach_path),
                "recomputed_restored_path_length_m": emi.path_length(restored_path),
                "minimal_plan_verified": episode["interactive_nav"][
                    "generation_validation"
                ]["minimal_plan_verified"],
            },
            "initial_state_application": initial_application,
            "camera_config": asdict(camera_config),
            "paths": {
                "initial_approach_path": np.asarray(approach_path, dtype=float).tolist(),
                "oracle_restored_path": np.asarray(restored_path, dtype=float).tolist(),
            },
            "pathpoints_topdown": pathpoint_path.name,
            "storyboard": storyboard_path.name,
            "story_steps": "story_steps.json",
            "steps": rows,
        }
        write_json(run_dir / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "output_dir": str(run_dir),
                    "episode_index": episode_index,
                    "case_id": annotations["case_id"],
                    "storyboard": storyboard_path.name,
                    "pathpoints_topdown": pathpoint_path.name,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    finally:
        if ctx is not None:
            probe.close_context(ctx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a representative required-door + Fridge mixed V3 episode and capture a "
            "five-step GT storyboard from a configurable right-rear shoulder camera."
        )
    )
    parser.add_argument("benchmark", type=Path, nargs="?", default=DEFAULT_BENCHMARK)
    parser.add_argument("--episode_index", type=int)
    parser.add_argument("--case_id")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--px_per_m", type=float, default=100.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--camera_name", default="mixed_story_shoulder_camera")
    parser.add_argument("--camera_behind_m", type=float, default=0.72)
    parser.add_argument("--camera_right_m", type=float, default=0.34)
    parser.add_argument("--camera_height_m", type=float, default=1.72)
    parser.add_argument("--camera_lookahead_m", type=float, default=1.75)
    parser.add_argument("--camera_look_height_m", type=float, default=1.08)
    parser.add_argument("--camera_fov_deg", type=float, default=62.0)
    parser.add_argument("--mujoco_gl", default="egl")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
