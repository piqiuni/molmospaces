#!/usr/bin/env python3
"""Run a reproducible 10-scene public HM3D-Sem v0.2 ObjectNav-v2 evaluation.

This runner uses the official Habitat Challenge 2023 task configuration and
the external navigation-only M2 policy.  It deliberately does not expose
Habitat semantic observations, goal geometry, or simulator state to the
policy.  Results and MLLM request evidence are written below ``/home/ldl``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--habitat-root", type=Path, default=Path("/home/ldl/habitat-objectnav/src/habitat-lab"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/ldl/habitat-objectnav/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/ldl/outputs/habitat_objectnav_v2_m2"))
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("/vepfs-wxy/memVLN/data/scene_datasets/hm3d/val/hm3d-val-habitat-v0.2"),
        help="authorized HM3D-Sem v0.2 val habitat asset root",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("/vepfs-wxy/memVLN/data/scene_datasets/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json"),
        help="matching authorized HM3D-Sem v0.2 annotated scene config",
    )
    parser.add_argument("--scene-count", type=int, default=10)
    parser.add_argument("--episodes-per-scene", type=int, default=1)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="repeatable HM3D scene directory name for a reproducible focused diagnostic, e.g. 00803-k1cupFYWXJ6",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--max-episode-seconds",
        type=int,
        default=500,
        help="wall-clock budget from the official v2 config; use 0 only for diagnostics",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mllm-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--mllm-model", default="qwen3.6-35b-a3b-fp8")
    parser.add_argument("--mllm-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=None,
        help="strict Habitat-v2 Module-1 detector / Module-2 navigation-only YAML profile",
    )
    parser.add_argument(
        "--validate-adapter-config",
        action="store_true",
        help="validate the adapter profile and official v2 invariants without constructing an Env or requesting models",
    )
    parser.add_argument(
        "--grounding-dino-endpoint",
        default="",
        help="optional loopback RGB-only GroundingDINO worker, e.g. http://127.0.0.1:12182",
    )
    parser.add_argument("--grounding-dino-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--yolov7-endpoint",
        default="",
        help="optional loopback COCO detector, e.g. http://127.0.0.1:12184; it creates track-only public RGB-D evidence",
    )
    parser.add_argument("--yolov7-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--mobile-sam-endpoint",
        default="",
        help="optional loopback RGB box-segmentation worker, e.g. http://127.0.0.1:12185; only refines detector-first public depth",
    )
    parser.add_argument("--mobile-sam-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--detector-first-target-tracking",
        action="store_true",
        help="allow optional YOLO evidence to seed/promote a public target standoff; Module-2 still selects the route and detector evidence never STOPs",
    )
    parser.add_argument(
        "--pointnav-endpoint",
        default="",
        help="optional loopback VLFM PointNav worker, e.g. http://127.0.0.1:12183",
    )
    parser.add_argument("--pointnav-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--collision-probe-forward-steps",
        type=int,
        default=0,
        help="public RGB-D side-probe steps after a no-motion recovery; 0 returns directly to M2 replanning",
    )
    parser.add_argument(
        "--frontier-revisit-cooldown-steps",
        type=int,
        default=0,
        help="optional public-map spatial cooldown for abandoned frontiers; disabled by default pending full-episode benefit",
    )
    parser.add_argument(
        "--clear-space-fallback-candidates",
        type=int,
        default=0,
        help="experimental M2-ranked observed-free fallback count; disabled by default after a negative long-horizon probe",
    )
    parser.add_argument(
        "--frontier-arrival-scan-steps",
        type=int,
        default=0,
        help="experimental public RGB-D scan turns after reaching a frontier, before the next M2 replan",
    )
    parser.add_argument(
        "--local-escape-relaxation-m",
        type=float,
        default=0.0,
        help="experimental raw-safe observed-free reconnect radius around public GPS when inflation fragments the map",
    )
    parser.add_argument(
        "--persistent-target-tracking",
        action="store_true",
        help="experimental public RGB-D target-surface tracker; validate on focused scenes before aggregate reporting",
    )
    parser.add_argument(
        "--public-trace",
        action="store_true",
        help="write opt-in per-action public-policy diagnostics for a focused run; normal metrics do not depend on it",
    )
    parser.add_argument(
        "--posthoc-step-metrics",
        action="store_true",
        help=(
            "write evaluator-only per-step official distance/success metrics after actions; "
            "they are never passed to the policy, Module-1, or Module-2"
        ),
    )
    parser.add_argument(
        "--posthoc-topdown-map",
        action="store_true",
        help=(
            "render an evaluator-only top-down navmesh with trajectory, official target centers, "
            "valid view points, target-perception evidence, and Module-2 target selections"
        ),
    )
    parser.add_argument(
        "--record-six-panel-video",
        action="store_true",
        help="record an evaluator-only 3x2 MP4 with RGB, depth, M1 detections, occupancy/route, semantic graph, and GT posthoc map",
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-frame-stride", type=int, default=1)
    parser.add_argument(
        "--allow-no-mllm-success",
        action="store_true",
        help="permit a diagnostic run even when no successful Module-2 selection is recorded",
    )
    args = parser.parse_args()
    if (
        args.scene_count < 1
        or args.episodes_per_scene < 1
        or args.max_steps < 1
        or args.max_episode_seconds < 0
        or args.clear_space_fallback_candidates < 0
        or args.frontier_arrival_scan_steps < 0
        or args.local_escape_relaxation_m < 0.0
        or args.video_fps <= 0.0
        or args.video_frame_stride < 1
    ):
        parser.error("scene/episode/step counts must be positive and timing/candidate counts cannot be negative")
    return args


def _ensure_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _select_episodes(
    episodes: list[Any],
    scene_count: int,
    episodes_per_scene: int,
    scene_ids: list[str] | None = None,
) -> list[Any]:
    by_scene: dict[str, list[Any]] = {}
    for episode in sorted(episodes, key=lambda item: (str(item.scene_id), str(item.episode_id))):
        by_scene.setdefault(str(episode.scene_id), []).append(episode)
    requested_scene_ids = sorted(set(scene_ids or []))
    if requested_scene_ids:
        # The episode scene path is rewritten to an absolute authorized asset
        # path before selection.  Its parent directory remains the canonical
        # HM3D scene ID used in dataset documentation and result files.
        paths_by_directory = {}
        for scene_path in by_scene:
            parent_name = Path(scene_path).parent.name
            paths_by_directory[parent_name or scene_path] = scene_path
        missing = [scene_id for scene_id in requested_scene_ids if scene_id not in paths_by_directory]
        if missing:
            raise ValueError(f"requested ObjectNav-v2 scene ID(s) not found: {', '.join(missing)}")
        scene_paths = [paths_by_directory[scene_id] for scene_id in requested_scene_ids]
    else:
        scene_paths = sorted(by_scene)[:scene_count]
    selected = []
    for scene_id in scene_paths:
        selected.extend(by_scene[scene_id][:episodes_per_scene])
    expected_scene_count = len(scene_paths)
    if len({str(item.scene_id) for item in selected}) != expected_scene_count:
        raise RuntimeError("failed to stratify selected evaluation episodes by unique scene")
    return selected


def _normalize_scene_dataset_config(episodes: list[Any], scene_dataset_config: Path) -> None:
    """Make v2 episode scene metadata independent of the process CWD."""

    normalized = str(scene_dataset_config)
    for episode in episodes:
        episode.scene_dataset_config = normalized


def _rewrite_scene_paths(episodes: list[Any], scene_root: Path) -> None:
    """Point v2 episodes at an authorized immutable HM3D asset layout.

    The public episode archive stores a Habitat-Lab-relative scene path.  This
    adapter intentionally resolves it to an absolute read-only asset path so
    there is no fragile local overlay or symlink layout to maintain.
    """

    for episode in episodes:
        source = Path(str(episode.scene_id))
        episode.scene_id = str(scene_root / source.parent.name / source.name)


def _public_episode(episode: Any, goal_category: str | None = None) -> dict[str, str]:
    return {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "object_category": str(goal_category or episode.object_category),
    }


_MLLM_CATEGORY_NAMES = {
    "plant": "potted plant",
    "sofa": "couch",
    "tv_monitor": "television",
}


def _mllm_goal_name(task_category: str) -> str:
    return _MLLM_CATEGORY_NAMES.get(task_category, task_category)


def _selection_manifest(selected: list[Any], episodes_per_scene: int) -> dict[str, Any]:
    """Persist the exact public evaluation slice before an episode runs."""

    return {
        "protocol": "deterministic scene-id-stratified slice of official ObjectNav HM3D v2 val",
        "scene_count": len({str(item.scene_id) for item in selected}),
        "episodes": len(selected),
        "episodes_per_scene": episodes_per_scene,
        "selection": [
            {
                **_public_episode(item),
                "start_position": [float(value) for value in item.start_position],
                "start_rotation": [float(value) for value in item.start_rotation],
            }
            for item in selected
        ],
    }


def _mllm_request_stats(path: Path) -> dict[str, Any]:
    """Summarize the append-only evidence emitted by the shared MLLM client."""

    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    by_role: dict[str, dict[str, int]] = {}
    for record in records:
        role = str(record.get("role", "unknown"))
        stats = by_role.setdefault(role, {"requests": 0, "successful": 0, "failed": 0})
        stats["requests"] += 1
        if record.get("error"):
            stats["failed"] += 1
        else:
            stats["successful"] += 1
    return {
        "requests": len(records),
        "successful": sum(stats["successful"] for stats in by_role.values()),
        "failed": sum(stats["failed"] for stats in by_role.values()),
        "by_role": by_role,
    }


def _fresh_mllm_requests_path(output_dir: Path) -> Path:
    """Preserve prior MLLM evidence before beginning a new evaluation run."""

    path = output_dir / "mllm_requests.jsonl"
    if not path.exists():
        return path
    archived = output_dir / "mllm_requests.previous.jsonl"
    suffix = 1
    while archived.exists():
        archived = output_dir / f"mllm_requests.previous.{suffix}.jsonl"
        suffix += 1
    path.replace(archived)
    return path


def main() -> int:
    args = _parse_args()
    _ensure_path(args.habitat_root / "habitat-lab")
    # Make ``python .../evaluate.py`` self-contained.  The package lives below
    # ``scripts/InteractiveNav``, not directly below ``scripts``.
    _ensure_path(Path(__file__).resolve().parents[1])
    from habitat_v2_adapter.adapter_config import (
        AdapterProfileError,
        load_adapter_profile,
        validate_habitat_v2_invariants,
    )

    profile = None
    if args.adapter_config is not None:
        try:
            profile = load_adapter_profile(args.adapter_config)
        except AdapterProfileError as exc:
            raise RuntimeError(f"invalid Habitat-v2 adapter profile: {exc}") from exc
    import habitat
    from habitat.config import read_write
    from habitat.datasets import make_dataset

    config = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml",
        overrides=[
            "habitat.dataset.split=val",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
            f"habitat.environment.max_episode_steps={args.max_steps}",
            f"habitat.environment.max_episode_seconds={args.max_episode_seconds}",
        ],
    )
    if profile is not None:
        try:
            validate_habitat_v2_invariants(config, profile)
        except AdapterProfileError as exc:
            raise RuntimeError(f"Habitat-v2 adapter profile invariant failed: {exc}") from exc
    if args.validate_adapter_config:
        if profile is None:
            raise RuntimeError("--validate-adapter-config requires --adapter-config")
        print(json.dumps(profile.public_summary(), indent=2, sort_keys=True))
        return 0
    # The default base directory may contain prior evidence.  Each invocation
    # gets an isolated child directory so metrics and JSONL logs are never
    # silently overwritten.
    run_stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    args.output_dir = args.output_dir / run_stamp
    suffix = 1
    while args.output_dir.exists():
        args.output_dir = args.output_dir.with_name(f"{run_stamp}-{suffix}")
        suffix += 1
    args.output_dir.mkdir(parents=True, exist_ok=False)

    from habitat_v2_adapter.policy import HabitatInteractiveNavM2Policy, PolicyConfig
    with read_write(config):
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = False
        # Assign after composition: raw `{split}` is valid Habitat template
        # syntax but is invalid when Hydra parses an override expression.
        config.habitat.dataset.data_path = str(
            args.data_root / "datasets/objectnav/hm3d/objectnav_hm3d_v2/{split}/{split}.json.gz"
        )
        config.habitat.dataset.scenes_dir = str(args.data_root / "scene_datasets")
    # ObjectNav-v2 episode JSON stores this path relative to the Habitat-Lab
    # checkout (``./data/...``).  The scenes themselves are deliberately kept
    # under /home/ldl, so normalize the per-episode field before Env constructs
    # its first Simulator instance.
    scene_dataset_config = args.scene_dataset_config
    if not scene_dataset_config.is_file():
        raise FileNotFoundError(
            "Missing authorized HM3D-Sem v0.2 scene configuration: "
            f"{scene_dataset_config}. Provide --scene-dataset-config for an "
            "authorized HM3D-Sem v0.2 val asset installation."
        )
    if not args.scene_root.is_dir():
        raise FileNotFoundError(
            "Missing authorized HM3D-Sem v0.2 habitat scene root: "
            f"{args.scene_root}. Provide --scene-root for an authorized installation."
        )
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    _rewrite_scene_paths(dataset.episodes, args.scene_root)
    _normalize_scene_dataset_config(dataset.episodes, scene_dataset_config)
    selected = _select_episodes(dataset.episodes, args.scene_count, args.episodes_per_scene, args.scene_id)
    missing_scenes = [str(item.scene_id) for item in selected if not Path(str(item.scene_id)).is_file()]
    if missing_scenes:
        preview = ", ".join(missing_scenes[:3])
        suffix = " ..." if len(missing_scenes) > 3 else ""
        raise FileNotFoundError(
            f"HM3D scene asset(s) missing for the selected slice: {preview}{suffix}. "
            "Check --scene-root against the authorized HM3D-Sem v0.2 installation."
        )
    env = habitat.Env(config=config, dataset=dataset)
    env.episodes = selected
    category_by_id = {int(category_id): category for category, category_id in dataset.category_to_task_category_id.items()}
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(_selection_manifest(selected, args.episodes_per_scene), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mllm_requests_path = _fresh_mllm_requests_path(args.output_dir)
    posthoc_metrics_path = args.output_dir / "posthoc_official_metrics.jsonl" if args.posthoc_step_metrics else None
    policy_values: dict[str, Any] = {
        "mllm_endpoint": args.mllm_endpoint,
        "mllm_model": args.mllm_model,
        "mllm_timeout_s": args.mllm_timeout_s,
        "grounding_dino_endpoint": args.grounding_dino_endpoint,
        "grounding_dino_timeout_s": args.grounding_dino_timeout_s,
        "yolov7_endpoint": args.yolov7_endpoint,
        "yolov7_timeout_s": args.yolov7_timeout_s,
        "detector_first_target_tracking": args.detector_first_target_tracking,
        "mobile_sam_endpoint": args.mobile_sam_endpoint,
        "mobile_sam_timeout_s": args.mobile_sam_timeout_s,
        "pointnav_endpoint": args.pointnav_endpoint,
        "pointnav_timeout_s": args.pointnav_timeout_s,
        "collision_probe_forward_steps": args.collision_probe_forward_steps,
        "frontier_revisit_cooldown_steps": args.frontier_revisit_cooldown_steps,
        "clear_space_fallback_candidates": args.clear_space_fallback_candidates,
        "frontier_arrival_scan_steps": args.frontier_arrival_scan_steps,
        "local_escape_relaxation_m": args.local_escape_relaxation_m,
        "persistent_target_tracking": (args.persistent_target_tracking or args.detector_first_target_tracking),
        "metrics_path": str(mllm_requests_path),
        "diagnostic_trace_path": (str(args.output_dir / "public_policy_trace.jsonl") if args.public_trace else ""),
    }
    sidecar_health: dict[str, Any] | None = None
    recorder_diagnostic_client = None
    if profile is not None:
        profile_values = profile.policy_overrides()
        policy_values.update(profile_values)
        # The profile is the source of truth for Module-1/2/3 policy settings.
        # Keep CLI switches as evaluator/data controls rather than silently
        # overriding the audited navigation-only bridge configuration.
        policy_values["metrics_path"] = str(mllm_requests_path)
        use_trace = bool(profile.raw["logging"]["public_trace"] or args.public_trace)
        policy_values["diagnostic_trace_path"] = str(args.output_dir / "public_policy_trace.jsonl") if use_trace else ""
        import yaml

        (args.output_dir / "effective_adapter_config.yaml").write_text(
            yaml.safe_dump(profile.raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        if bool(profile.module1["sidecar"]["require_healthy"]):
            from habitat_v2_adapter.module1_bridge import FullRosStackClient, Module1DetectorSidecarClient

            if profile.module1["mode"] == "full_ros_navigation_stack":
                probe = FullRosStackClient(
                    endpoint=str(profile.module1["sidecar"]["endpoint"]),
                    timeout_s=float(profile.module1["sidecar"]["timeout_s"]),
                )
                recorder_diagnostic_client = probe
            else:
                probe = Module1DetectorSidecarClient(
                    endpoint=str(profile.module1["sidecar"]["endpoint"]),
                    timeout_s=float(profile.module1["sidecar"]["timeout_s"]),
                    include_depth=bool(profile.module1["sidecar"]["include_depth"]),
                    original_module1_scripts=str(profile.module1["sidecar"]["original_module1_scripts"]),
                )
            sidecar_health = probe.health()
    policy = HabitatInteractiveNavM2Policy(PolicyConfig(**policy_values))
    rows: list[dict[str, Any]] = []
    decision_stats = {"decisions": 0, "model_selected": 0, "fallback_selected": 0}
    vision_stats = {
        "vision_queries": 0,
        "vision_failed": 0,
        "vision_positive": 0,
        "vision_temporally_confirmed": 0,
        "visual_controller_steps": 0,
        "visual_stop_emitted": 0,
        "visual_goal_clearance_blocks": 0,
        "visual_goal_releases": 0,
        "visual_goal_budget_releases": 0,
        "grounding_dino_queries": 0,
        "grounding_dino_failed": 0,
        "grounding_dino_confirmed": 0,
        "module1_detector_queries": 0,
        "module1_detector_failed": 0,
        "module1_detector_detections": 0,
        "module1_detector_positive": 0,
        "full_ros_queries": 0,
        "full_ros_failed": 0,
        "full_ros_graph_nodes": 0,
        "full_ros_graph_edges": 0,
        "yolov7_queries": 0,
        "yolov7_failed": 0,
        "yolov7_positive": 0,
        "mobile_sam_queries": 0,
        "mobile_sam_failed": 0,
        "mobile_sam_confirmed": 0,
        "target_track_seeded": 0,
        "target_track_promoted": 0,
        "target_track_updated": 0,
        "target_track_rejected": 0,
        "target_track_expired": 0,
        "m3_queries": 0,
        "m3_stop_emitted": 0,
        "m3_failed": 0,
    }
    local_control_stats = {
        "pointnav_queries": 0,
        "pointnav_failed": 0,
        "pointnav_forward_steps": 0,
        "pointnav_turn_steps": 0,
        "pointnav_stop_predictions": 0,
        "pointnav_safety_blocks": 0,
        "pointnav_macro_aborts": 0,
        "pointnav_aborted_micro_actions": 0,
    }
    try:
        for expected_episode in selected:
            observations = env.reset()
            current_episode = env.current_episode
            if (
                str(current_episode.scene_id) != str(expected_episode.scene_id)
                or str(current_episode.object_category) != str(expected_episode.object_category)
                or not all(
                    abs(float(actual) - float(expected)) <= 1e-6
                    for actual, expected in zip(current_episode.start_position, expected_episode.start_position)
                )
            ):
                raise RuntimeError("environment reset did not match the persisted selected evaluation episode")
            goal_id = int(observations["objectgoal"][0])
            public_episode = _public_episode(
                env.current_episode,
                _mllm_goal_name(category_by_id[goal_id]),
            )
            policy.reset(public_episode)
            steps = 0
            posthoc_first_distance: float | None = None
            posthoc_min_distance: float | None = None
            posthoc_min_step: int | None = None
            posthoc_episode_rows: list[dict[str, Any]] = []
            video_recorder = None
            if args.record_six_panel_video:
                from habitat_v2_adapter.six_panel_video import SixPanelVideoRecorder

                scene_name = Path(str(current_episode.scene_id)).parent.name
                video_recorder = SixPanelVideoRecorder(
                    path=args.output_dir / f"six_panel_{scene_name}_ep{current_episode.episode_id}.mp4",
                    env=env,
                    episode=current_episode,
                    fps=args.video_fps,
                    frame_stride=args.video_frame_stride,
                )
            try:
                for steps in range(1, args.max_steps + 1):
                    action = policy.act(observations)
                    policy.assert_navigation_only(action)
                    if video_recorder is not None:
                        # Record the exact public RGB-D frame consumed by the
                        # policy. Recording after env.step() would overlay the
                        # previous detector result on the next camera frame.
                        video_metrics = dict(env.get_metrics())
                        video_state = env.sim.get_agent_state()
                        if recorder_diagnostic_client is not None:
                            diagnostic_panel = video_recorder.render_recorder_topdown(
                                policy=policy,
                                metrics=video_metrics,
                                agent_world_position=[float(value) for value in video_state.position],
                            )
                            diagnostic_error = recorder_diagnostic_client.publish_recorder_diagnostic(
                                diagnostic_panel[..., ::-1],
                                step=steps,
                            )
                            if diagnostic_error:
                                raise RuntimeError(
                                    f"could not publish evaluator-only recorder panel 6: {diagnostic_error}"
                                )
                        video_recorder.append(
                            observations=observations,
                            action=action,
                            policy=policy,
                            metrics=video_metrics,
                            agent_world_position=[float(value) for value in video_state.position],
                        )
                    observations = env.step(action)
                    if posthoc_metrics_path is not None:
                        # This is intentionally post-action evaluator instrumentation.
                        # The policy has already emitted its action and receives none
                        # of these official task measures or global simulator fields.
                        step_metrics = dict(env.get_metrics())
                        raw_distance = step_metrics.get("distance_to_goal")
                        try:
                            distance_to_goal = float(raw_distance) if raw_distance is not None else None
                        except (TypeError, ValueError):
                            distance_to_goal = None
                        if distance_to_goal is not None:
                            if posthoc_first_distance is None:
                                posthoc_first_distance = distance_to_goal
                            if posthoc_min_distance is None or distance_to_goal < posthoc_min_distance:
                                posthoc_min_distance = distance_to_goal
                                posthoc_min_step = steps
                        agent_state = env.sim.get_agent_state()
                        posthoc_row = {
                            "event": "posthoc_official_metrics",
                            **public_episode,
                            "step": steps,
                            "action": str(action.get("action")),
                            "episode_over": bool(env.episode_over),
                            "distance_to_goal": distance_to_goal,
                            "success": float(step_metrics.get("success", 0.0)),
                            "spl": float(step_metrics.get("spl", 0.0)),
                            # This simulator pose is recorded strictly for offline
                            # top-down rendering after the action; it never crosses
                            # the evaluator-to-policy observation boundary.
                            "agent_world_position": [float(value) for value in agent_state.position],
                        }
                        with posthoc_metrics_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(posthoc_row, sort_keys=True) + "\n")
                        posthoc_episode_rows.append(posthoc_row)
                    if env.episode_over:
                        break
            finally:
                if video_recorder is not None:
                    video_recorder.close()
            metrics = dict(env.get_metrics())
            episode_decisions = policy.decision_stats()
            episode_vision = policy.vision_stats()
            episode_local_control = policy.local_control_stats()
            if args.posthoc_topdown_map:
                if posthoc_metrics_path is None:
                    raise RuntimeError("--posthoc-topdown-map requires --posthoc-step-metrics")
                from habitat_v2_adapter.posthoc_topdown import render_posthoc_topdown

                render_posthoc_topdown(
                    output_dir=args.output_dir,
                    env=env,
                    episode=current_episode,
                    public_episode=public_episode,
                    posthoc_rows=posthoc_episode_rows,
                    trace_path=Path(policy.config.diagnostic_trace_path),
                    episode_vision=episode_vision,
                    metrics=metrics,
                )
            decision_stats["decisions"] += episode_decisions["decisions"]
            decision_stats["model_selected"] += episode_decisions["model_selected"]
            decision_stats["fallback_selected"] += episode_decisions["fallback_selected"]
            for key in vision_stats:
                vision_stats[key] += episode_vision[key]
            for key in local_control_stats:
                local_control_stats[key] += episode_local_control[key]
            rows.append(
                {
                    **public_episode,
                    "steps": steps,
                    "mllm_decisions": episode_decisions["decisions"],
                    "mllm_model_selected": episode_decisions["model_selected"],
                    "mllm_fallback_selected": episode_decisions["fallback_selected"],
                    **episode_vision,
                    **episode_local_control,
                    "posthoc_first_distance_to_goal": posthoc_first_distance,
                    "posthoc_min_distance_to_goal": posthoc_min_distance,
                    "posthoc_min_distance_step": posthoc_min_step,
                    **metrics,
                }
            )
    finally:
        env.close()

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    scene_rows = []
    for scene_id, scene_episodes in sorted(by_scene.items()):
        scene_rows.append(
            {
                "scene_id": scene_id,
                "episodes": len(scene_episodes),
                "success": sum(float(item.get("success", 0.0)) for item in scene_episodes) / len(scene_episodes),
                "spl": sum(float(item.get("spl", 0.0)) for item in scene_episodes) / len(scene_episodes),
            }
        )
    mllm_stats = _mllm_request_stats(mllm_requests_path)
    subgoal_successes = mllm_stats["by_role"].get("subgoal_selection", {}).get("successful", 0)
    if not args.allow_no_mllm_success and (
        not subgoal_successes or not decision_stats["model_selected"]
    ):
        raise RuntimeError(
            "no successful Module-2 model-backed subgoal selection was recorded"
        )
    summary = {
        "protocol": "Habitat Challenge 2023 ObjectNav-v2 / HM3D-Sem v0.2 val / Stretch continuous control",
        "policy": HabitatInteractiveNavM2Policy.name,
        "navigation_only": True,
        "uses_oracle_gt": False,
        "scene_count": len(scene_rows),
        "episodes": len(rows),
        "episodes_per_scene": args.episodes_per_scene,
        "max_episode_steps": args.max_steps,
        "max_episode_seconds": args.max_episode_seconds,
        "scene_root": str(args.scene_root),
        "scene_dataset_config": str(scene_dataset_config),
        "goal_category_source": "public ObjectGoalSensor task-category ID",
        "adapter_profile": profile.public_summary() if profile is not None else None,
        "module1_sidecar_health": sidecar_health,
        "persistent_target_tracking": bool(
            policy.config.persistent_target_tracking
        ),
        "module1_detector_only": bool(policy.config.module1_detector_enabled),
        "module3_enabled": bool(policy.config.module3_enabled),
        "detector_first_target_tracking": bool(
            policy.config.module1_detector_enabled or args.detector_first_target_tracking
        ),
        "yolov7_detector": bool(args.yolov7_endpoint),
        "pointnav_local_controller": bool(args.pointnav_endpoint),
        "collision_probe_forward_steps": args.collision_probe_forward_steps,
        "frontier_revisit_cooldown_steps": args.frontier_revisit_cooldown_steps,
        "clear_space_fallback_candidates": args.clear_space_fallback_candidates,
        "frontier_arrival_scan_steps": args.frontier_arrival_scan_steps,
        "local_escape_relaxation_m": args.local_escape_relaxation_m,
        "public_trace": bool(policy.config.diagnostic_trace_path),
        "posthoc_step_metrics": bool(posthoc_metrics_path is not None),
        "posthoc_topdown_map": bool(args.posthoc_topdown_map),
        "six_panel_video": bool(args.record_six_panel_video),
        "video_fps": float(args.video_fps) if args.record_six_panel_video else None,
        "video_frame_stride": int(args.video_frame_stride) if args.record_six_panel_video else None,
        "mllm_requests": mllm_stats,
        "mllm_policy_decisions": decision_stats,
        "vision_policy_stats": vision_stats,
        "local_control_stats": local_control_stats,
        "mean_success": sum(float(item.get("success", 0.0)) for item in rows) / max(1, len(rows)),
        "mean_spl": sum(float(item.get("spl", 0.0)) for item in rows) / max(1, len(rows)),
        "scenes": scene_rows,
    }
    (args.output_dir / "episodes.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
