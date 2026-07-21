"""Record one channel or container V3 interaction as a full rollout.

The mixed runner remains the reference for chained door->container episodes;
this entry point uses the same force/policy executor and H5 recorder for the
two standalone domains so the unified collector has one full-mode contract.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import capture_mixed_gt_storyboard as storyboard
from scripts.InteractiveNav import interactive_nav_v3
from scripts.InteractiveNav import record_mixed_rby1_rollout as mixed
from scripts.InteractiveNav import visualize_mixed_interaction_benchmark as mixed_viz
from scripts.InteractiveNav.collection.full_rollout_recorder import (
    H5StepRolloutRecorder,
    validate_full_rollout,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mixed.probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def _single_annotation(episode: dict, domain: str) -> dict:
    interactions = list(episode["interactive_nav"]["interactions"])
    if domain == "channel":
        row = next(item for item in interactions if str(item["type"]).startswith("channel_"))
        return {
            "kind": "door",
            "target_name": row["object_name"],
            "joint_index": int(row["joint_index"]),
            "interaction_id": row["interaction_id"],
            "door_root": row.get("door_root_name", row["object_name"]),
        }
    row = next(item for item in interactions if str(item["type"]).startswith("container_"))
    return {
        "kind": "container",
        "target_name": row["object_name"],
        "joint_index": int(row["joint_index"]),
        "interaction_id": row["interaction_id"],
        "door_root": "",
    }


def _choose_episode(
    episodes: list[dict], *, episode_index: int | None, case_id: str | None, domain: str
) -> tuple[int, dict, dict]:
    """Select and validate a standalone channel/container episode."""
    validator = {
        "channel": interactive_nav_v3.validate_channel_v3_episode,
        "container": interactive_nav_v3.validate_container_v3_episode,
    }[domain]
    if episode_index is not None:
        if not 0 <= episode_index < len(episodes):
            raise ValueError(f"Episode index out of range: {episode_index}")
        episode = episodes[episode_index]
        validator(episode)
        return episode_index, episode, {"selection_mode": "explicit_episode_index"}
    if case_id is not None:
        for index, episode in enumerate(episodes):
            if episode.get("interactive_nav", {}).get("case_id") == case_id:
                validator(episode)
                return index, episode, {"selection_mode": "explicit_case_id"}
        raise ValueError(f"{domain} case_id not found: {case_id}")
    raise ValueError("Standalone full rollout requires --case_id or --episode_index")


def _post_navigation_goal(episode: dict, interaction_id: str) -> list[float] | None:
    plans = episode.get("interactive_nav", {}).get("oracle_plans", [])
    for plan in plans:
        opened = False
        for step in plan.get("steps", []):
            if step.get("type") == "open_joint" and step.get("interaction_id") == interaction_id:
                opened = True
                continue
            if opened and step.get("type") == "navigate" and step.get("interaction_id") is None:
                goal = list(step["goal_point"])
                yaw = float(step.get("goal_yaw", 0.0))
                quat = mixed.R.from_euler("Z", yaw).as_quat(scalar_first=True)
                return [
                    float(goal[0]),
                    float(goal[1]),
                    float(goal[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ]
    return None


def run(args: argparse.Namespace) -> int:
    episodes = storyboard.load_episodes(args.benchmark)
    episode_index, episode, selection = _choose_episode(
        episodes,
        episode_index=args.episode_index,
        case_id=args.case_id,
        domain=args.domain,
    )
    domains = episode["interactive_nav"]["interaction_domains"]
    if domains != [args.domain]:
        raise ValueError(f"Episode domains {domains!r} do not match --domain {args.domain!r}")
    spec = _single_annotation(episode, args.domain)
    case_id = str(episode["interactive_nav"]["case_id"])
    run_dir = args.output_dir / f"episode_{episode_index:04d}_{args.domain}_{storyboard.safe_slug(case_id, 96)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    operation_pose = mixed.oracle_operation_pose(episode, spec["interaction_id"])
    if operation_pose is None:
        raise RuntimeError(f"No oracle operation pose for {spec['interaction_id']}")
    start_pose = list(episode["task"]["robot_base_pose"])
    operation_spec, target_meta, _ = mixed.prepare_operation_spec(
        episode,
        house_index=int(episode["house_index"]),
        interaction_kind=spec["kind"],
        target_name=spec["target_name"],
        joint_index=spec["joint_index"],
        start_pose=start_pose,
        operation_pose_override=operation_pose,
        args=args,
    )
    path, path_length = mixed.compute_navigation_path(
        episode,
        start_xy=np.asarray(start_pose[:2], dtype=float),
        goal_xy=np.asarray(operation_pose[:2], dtype=float),
        door_state="closed",
        required_door_root=spec["door_root"],
        args=args,
    )
    if path:
        path[-1] = mixed.pos_quat_to_pose_mat(operation_pose)
    post_path: list[np.ndarray] = []
    post_goal = _post_navigation_goal(episode, spec["interaction_id"])
    if post_goal is not None:
        post_path, _post_path_length = mixed.compute_navigation_path(
            episode,
            start_xy=np.asarray(operation_pose[:2], dtype=float),
            goal_xy=np.asarray(post_goal[:2], dtype=float),
            door_state="open" if spec["kind"] == "door" else "closed",
            required_door_root=spec["door_root"],
            args=args,
        )
        if post_path:
            post_path[-1] = mixed.pos_quat_to_pose_mat(post_goal)

    recorder = H5StepRolloutRecorder(
        run_dir / "trajectory.h5",
        episode_id=case_id,
        camera_names=mixed.CAMERAS,
        metadata={
            "schema_version": "interactive_nav_full_rollout_v1",
            "benchmark": str(args.benchmark),
            "episode_index": episode_index,
            "house_index": int(episode["house_index"]),
            "interaction_domains": [args.domain],
        },
    )
    step_collector = mixed.StepCollector(recorder)
    unfinished = True

    def abort_unfinished() -> None:
        if unfinished and not recorder.closed:
            recorder.abort("process_exited_before_rollout_finalize")

    atexit.register(abort_unfinished)
    output = run_dir / "interaction"
    try:
        result = mixed.probe.execute_rby1_whole_body_interaction(
            mixed.probe.build_rby1_interaction_config(
                mixed.request_args(
                    house_index=int(episode["house_index"]),
                    interaction_kind=spec["kind"],
                    target_name=spec["target_name"],
                    joint_index=spec["joint_index"],
                    args=args,
                )
            ),
            operation_spec,
            interaction_kind=spec["kind"],
            variant=args.variant,
            output_dir=output,
            camera_names=mixed.CAMERAS,
            max_steps=args.max_steps,
            video_fps=args.video_fps,
            base_adjustment_path=path,
            max_base_adjustment_steps=max(
                args.max_base_adjustment_steps, 5 * len(path) + 10
            ),
            post_interaction_path=post_path,
            max_post_interaction_steps=max(args.max_steps, 5 * len(post_path) + 10),
            lock_base_during_force=args.lock_base_during_force,
            initial_state_episode=episode,
            step_callback=step_collector.callback(f"nav_to_{args.domain}_and_open"),
            interaction_executor=args.interaction_executor,
            allow_force_fallback=args.allow_force_fallback,
            force_fallback_target_fraction=mixed.force_target_fraction(
                args, "door" if spec["kind"] == "door" else "container"
            ),
            force_fallback_max_steps=mixed.force_max_steps(
                args, "door" if spec["kind"] == "door" else "container"
            ),
            completion_hold_seconds=args.completion_hold_seconds,
        )
        write_json(output / "result.json", result)
        success = bool(result.get("success")) and mixed.semantic_fraction(result) >= args.required_open_fraction
        recorder.finalize(
            success=success,
            terminal_reason="interaction_completed" if success else "interaction_failed",
            result=result,
        )
        unfinished = False
        audit = validate_full_rollout(run_dir / "trajectory.h5")
        write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "interactive_nav_full_single_rollout_v1",
                "benchmark": str(args.benchmark),
                "episode_index": episode_index,
                "case_id": case_id,
                "domain": args.domain,
                "selection": selection,
                "target_meta": target_meta,
                "path_length_m": path_length,
                "waypoint_count": len(path),
                "result": result,
                "trajectory_path": str(run_dir / "trajectory.h5"),
                "trajectory_audit": audit,
            },
        )
        print(json.dumps({"output_dir": str(run_dir), "trajectory_audit": audit}, ensure_ascii=False))
        return 0 if success else 1
    finally:
        if unfinished:
            abort_unfinished()
        atexit.unregister(abort_unfinished)


def build_parser() -> argparse.ArgumentParser:
    parser = mixed.build_parser()
    parser.description = "Run one standalone V3 channel or container full rollout."
    parser.add_argument("--domain", choices=["channel", "container"], required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
