#!/usr/bin/env python3
"""Render top-down reports for completed native ``nav_to_obj`` batch attempts.

The native evaluator stores one ``batch_result.json`` per isolated attempt, but
its artifact layout differs from the InteractiveNav V3 result format consumed by
``episode_topdown.py``.  This adapter keeps the established static-map renderer
and converts only post-hoc artifacts:

* the frozen benchmark locates the robot start and eligible task instances;
* GT frame manifests locate candidates that became visible during the run;
* sparse ``Action command[base]`` entries and an official H5 endpoint form a
  clearly-labelled, non-continuous trace;
* ``official_nav_to_obj_result.json`` supplies the official terminal outcome.

No simulator, ROS node, or policy process is started.  It is safe to run while
the batch manager continues evaluating other episodes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav.evaluation.episode_topdown import render_episode_topdown


SCHEMA_VERSION = "native_nav_to_obj_batch_topdown_v1"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ACTION_PATTERN = re.compile(
    rf"Step:\s*(?P<step>\d+)\s+Action command\[base\]:\s*\[\s*"
    rf"(?P<x>{_NUMBER})\s+(?P<y>{_NUMBER})\s+(?P<yaw>{_NUMBER})\s*\]",
    flags=re.MULTILINE,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _select_terminal_attempts(
    run_root: Path,
    *,
    allowed_statuses: set[str],
    selected_indices: set[int] | None,
) -> list[tuple[int, Path, dict[str, Any]]]:
    """Choose the final committed attempt per episode from a batch run root."""

    selected: dict[int, tuple[tuple[int, float, str], Path, dict[str, Any]]] = {}
    for path in sorted(run_root.glob("episodes/episode_*/attempt_*/batch_result.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or str(payload.get("status", "")) not in allowed_statuses:
            continue
        claim = payload.get("claim")
        claim = claim if isinstance(claim, dict) else {}
        episode_index = _as_int(claim.get("episode_idx"))
        if episode_index is None or (selected_indices is not None and episode_index not in selected_indices):
            continue
        attempt = _as_int(claim.get("attempt")) or 0
        finished = _as_float(payload.get("finished_at")) or 0.0
        key = (attempt, finished, str(path))
        current = selected.get(episode_index)
        if current is None or key > current[0]:
            selected[episode_index] = (key, path, payload)
    return [
        (episode_index, path, payload)
        for episode_index, (_key, path, payload) in sorted(selected.items())
    ]


def _parse_sparse_action_trace(stdout_path: Path) -> list[dict[str, Any]]:
    """Recover sparse base poses emitted by the native evaluator's progress log."""

    if not stdout_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    previous: tuple[float, float, float] | None = None
    for match in _ACTION_PATTERN.finditer(stdout_path.read_text(encoding="utf-8", errors="replace")):
        pose = (float(match["x"]), float(match["y"]), float(match["yaw"]))
        if previous is not None and all(math.isclose(a, b, abs_tol=1e-7) for a, b in zip(pose, previous)):
            continue
        rows.append(
            {
                "step_index": int(match["step"]),
                "base": {"base_pose_xyyaw": list(pose)},
            }
        )
        previous = pose
    return rows


def _load_official_result(debug_dir: Path) -> dict[str, Any]:
    candidates = [debug_dir / "official_nav_to_obj_result.json"]
    candidates.extend(sorted(debug_dir.glob("episode_*/official_nav_to_obj_result.json")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _load_target_selection(debug_dir: Path) -> tuple[dict[str, Any], Path | None]:
    """Read the native evaluator's frozen eligible-instance set when present."""

    paths = [debug_dir / "target_selection.json", *sorted(debug_dir.glob("episode_*/target_selection.json"))]
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload, path
    return {}, None


def _candidate_names(target_selection: dict[str, Any], target_name: str | None) -> list[str]:
    """Preserve native any-candidate semantics while keeping deterministic order."""

    raw_candidates = target_selection.get("candidate_instances")
    raw_candidates = raw_candidates if isinstance(raw_candidates, list) else []
    names: list[str] = []
    for raw in raw_candidates:
        name = raw if isinstance(raw, str) else None
        if name and name not in names:
            names.append(name)
    if target_name and target_name not in names:
        names.insert(0, target_name)
    return names


def _benchmark_candidate_names(benchmark_episode: dict[str, Any], target_selection: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Return the official designated candidate and every valid-success instance."""

    task = benchmark_episode.get("task")
    task = task if isinstance(task, dict) else {}
    target_name = task.get("pickup_obj_name")
    if not isinstance(target_name, str) or not target_name:
        return None, []
    task_candidates = task.get("pickup_obj_candidates")
    task_candidates = task_candidates if isinstance(task_candidates, list) else []
    normalized_selection = dict(target_selection)
    if not normalized_selection.get("candidate_instances"):
        normalized_selection["candidate_instances"] = task_candidates
    return target_name, _candidate_names(normalized_selection, target_name)


def _candidate_context_from_gt_frames(
    manifest_path: Path,
    *,
    candidate_names: list[str],
    target_name: str | None,
) -> list[dict[str, Any]]:
    """Locate candidates from evaluator-produced world-frame GT observations.

    NavToObj targets may be unchanged scene objects and therefore absent from
    ``scene_modifications.object_poses``.  The policy-debug frame manifest is
    post-hoc evaluator output and contains their world-frame 3D boxes whenever
    they were observed.  Pick the highest-pixel observation for each instance.
    """

    if not manifest_path.is_file() or not candidate_names:
        return []
    candidate_set = set(candidate_names)
    best: dict[str, tuple[int, int, dict[str, Any]]] = {}
    try:
        lines = manifest_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(frame, dict):
            continue
        step_index = _as_int(frame.get("step_index")) or 0
        observations = frame.get("gt_observations")
        observations = observations if isinstance(observations, dict) else {}
        rows = observations.get("observations")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            object_name = row.get("id")
            if not isinstance(object_name, str) or object_name not in candidate_set:
                continue
            box_3d = row.get("box_3d")
            box_3d = box_3d if isinstance(box_3d, dict) else {}
            center = box_3d.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            x, y = _as_float(center[0]), _as_float(center[1])
            if x is None or y is None:
                continue
            visible_pixels = _as_int(row.get("visible_pixels")) or 0
            marker = {
                "xy": [x, y],
                "object_name": object_name,
                "is_designated": object_name == target_name,
                "source": "evaluator_gt_frame_observation",
                "observed_step_index": step_index,
                "visible_pixels": visible_pixels,
                "visible_fraction": _as_float(row.get("visible_fraction")),
            }
            current = best.get(object_name)
            if current is None or (visible_pixels, step_index) > current[:2]:
                best[object_name] = (visible_pixels, step_index, marker)
    return [best[name][2] for name in candidate_names if name in best]


def _target_context_from_candidates(
    candidates: list[dict[str, Any]], target_name: str | None
) -> dict[str, Any] | None:
    """Use the designated candidate only as a reference, never as completion proof."""

    if target_name is None:
        return None
    for candidate in candidates:
        if candidate.get("object_name") != target_name:
            continue
        return {
            **candidate,
            "label": "designated benchmark candidate",
            "source": "evaluator_gt_frame_observation",
        }
    return None


def _mark_inferred_success_candidate(
    candidates: list[dict[str, Any]],
    *,
    official_success: Any,
    terminal_pose: dict[str, Any] | None,
    benchmark_episode: dict[str, Any],
) -> list[dict[str, Any]]:
    """Highlight one candidate only when the official result supports the inference.

    Native evaluator outputs do not retain the winning object id.  On a formal
    success, the terminal H5 pose plus the frozen success-radius lets the report
    identify the nearest *observed* eligible instance as an inference, not a
    ground-truth completion record.
    """

    if official_success is not True or terminal_pose is None or not candidates:
        return candidates
    pose = terminal_pose.get("xyyaw")
    if not isinstance(pose, list) or len(pose) < 2:
        return candidates
    x, y = _as_float(pose[0]), _as_float(pose[1])
    if x is None or y is None:
        return candidates
    task = benchmark_episode.get("task")
    task = task if isinstance(task, dict) else {}
    threshold = _as_float(task.get("succ_pos_threshold"))
    if threshold is None or threshold <= 0.0:
        return candidates
    nearest = min(
        candidates,
        key=lambda marker: math.hypot(float(marker["xy"][0]) - x, float(marker["xy"][1]) - y),
    )
    distance = math.hypot(float(nearest["xy"][0]) - x, float(nearest["xy"][1]) - y)
    if distance > threshold + 1e-6:
        return candidates
    marked: list[dict[str, Any]] = []
    for marker in candidates:
        copied = dict(marker)
        if marker is nearest:
            copied.update(
                {
                    "is_official_success_candidate_inferred": True,
                    "terminal_distance_m": distance,
                    "success_distance_threshold_m": threshold,
                    "inference": "official_success_plus_terminal_h5_within_frozen_threshold",
                }
            )
        marked.append(copied)
    return marked


def _terminal_pose_from_h5(attempt_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Read the official evaluator's terminal base pose without replaying a task."""

    h5_paths = sorted(attempt_dir.glob("output/**/*.h5"))
    if not h5_paths:
        return None, None
    try:
        import h5py
    except ImportError:
        return None, None
    for path in h5_paths:
        try:
            with h5py.File(path, "r") as handle:
                dataset = handle.get("traj_0/obs/extra/robot_base_pose")
                if dataset is None or not len(dataset):
                    continue
                values = [float(value) for value in dataset[-1].tolist()]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(values) < 2:
            continue
        x, y = _as_float(values[0]), _as_float(values[1])
        if x is None or y is None:
            continue
        yaw = 0.0
        if len(values) >= 7:
            qw, qz = _as_float(values[3]), _as_float(values[6])
            if qw is not None and qz is not None:
                yaw = 2.0 * math.atan2(qz, qw)
        return {"xyyaw": [x, y, yaw], "source": "official_h5_terminal_pose"}, path
    return None, None


def _append_terminal_pose(trace: list[dict[str, Any]], terminal_pose: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Ensure the red endpoint represents the official evaluator state, if saved."""

    if terminal_pose is None:
        return trace
    pose = terminal_pose.get("xyyaw")
    if not isinstance(pose, list) or len(pose) < 3:
        return trace
    rows = list(trace)
    next_step = max((_as_int(row.get("step_index")) or 0 for row in rows), default=0) + 1
    rows.append({"step_index": next_step, "base": {"base_pose_xyyaw": pose}, "source": terminal_pose["source"]})
    return rows


def _trajectory_source(trace: list[dict[str, Any]], terminal_pose: dict[str, Any] | None) -> str:
    if terminal_pose is None:
        return "sparse_stdout_action_trace" if trace else "unavailable"
    return "sparse_stdout_action_trace_plus_terminal_h5_pose" if trace else "official_h5_terminal_pose"


def _outcome_label(batch_result: dict[str, Any]) -> str:
    """Keep formal navigation outcomes separate from no-rollout exclusions."""

    if str(batch_result.get("status")) == "completed":
        return "official_success" if batch_result.get("official_success") is True else "official_failure"
    error = str(batch_result.get("error_message") or "").lower()
    if batch_result.get("return_code") == 0 and "total_count is not 1" in error:
        return "unevaluable_asset_or_sampling"
    return "execution_failure"


def _native_adapter_document(
    *,
    episode_index: int,
    batch_result: dict[str, Any],
    official_result: dict[str, Any],
    trace: list[dict[str, Any]],
    outcome: str,
    trajectory_source: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episode_index": episode_index,
        "status": outcome,
        "terminal_reason": outcome,
        "official_success": batch_result.get("official_success"),
        "source": "native_nav_to_obj_batch_adapter",
        "topdown_title": "Native NavToObj batch top-down",
        "trajectory_source": trajectory_source,
    }
    for key in ("task_description", "distance_m", "head_camera_visible", "horizon"):
        if key in official_result:
            result[key] = official_result[key]
    return {"result": result, "trace": trace}


def render_batch(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    benchmark_path = args.benchmark.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    benchmark_file = benchmark_path / "benchmark.json" if benchmark_path.is_dir() else benchmark_path
    benchmark = _read_json(benchmark_file)
    if not isinstance(benchmark, list):
        raise ValueError(f"Expected a benchmark JSON list: {benchmark_file}")
    selected_indices = None if args.episode_indices is None else set(args.episode_indices)
    allowed_statuses = {part.strip() for part in args.statuses.split(",") if part.strip()}
    attempts = _select_terminal_attempts(
        run_root,
        allowed_statuses=allowed_statuses,
        selected_indices=selected_indices,
    )
    if args.limit is not None:
        attempts = attempts[: args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name
    records: list[dict[str, Any]] = []
    for episode_index, result_path, batch_result in attempts:
        attempt_dir = result_path.parent
        outcome = _outcome_label(batch_result)
        image_path = output_dir / f"episode_{episode_index:08d}_{outcome}.png"
        metadata_path = image_path.with_suffix(".json")
        record: dict[str, Any] = {
            "episode_index": episode_index,
            "outcome": outcome,
            "batch_result": str(result_path),
            "image": str(image_path),
            "metadata": str(metadata_path),
        }
        if not 0 <= episode_index < len(benchmark):
            record["status"] = "skipped_invalid_benchmark_index"
            records.append(record)
            _write_json(manifest_path, _manifest_payload(run_root, benchmark_file, records))
            continue
        if image_path.is_file() and metadata_path.is_file() and not args.overwrite:
            record["status"] = "skipped_existing"
            records.append(record)
            _write_json(manifest_path, _manifest_payload(run_root, benchmark_file, records))
            continue
        debug_dir = attempt_dir / "debug"
        sparse_trace = _parse_sparse_action_trace(attempt_dir / "stdout.log")
        terminal_pose, terminal_pose_path = _terminal_pose_from_h5(attempt_dir)
        trace = _append_terminal_pose(sparse_trace, terminal_pose)
        trajectory_source = _trajectory_source(sparse_trace, terminal_pose)
        official_result = _load_official_result(debug_dir)
        target_selection, target_selection_path = _load_target_selection(debug_dir)
        target_name, candidate_names = _benchmark_candidate_names(benchmark[episode_index], target_selection)
        frame_manifest = attempt_dir / "sim_step_frames" / "manifest.jsonl"
        candidate_context = _candidate_context_from_gt_frames(
            frame_manifest,
            candidate_names=candidate_names,
            target_name=target_name,
        )
        candidate_context = _mark_inferred_success_candidate(
            candidate_context,
            official_success=batch_result.get("official_success"),
            terminal_pose=terminal_pose,
            benchmark_episode=benchmark[episode_index],
        )
        target_context = _target_context_from_candidates(candidate_context, target_name)
        adapter_dir = output_dir / "adapters" / f"episode_{episode_index:08d}"
        adapter_result_path = adapter_dir / "episode_result.json"
        adapter_context_path = adapter_dir / "episode_visualization.json"
        _write_json(
            adapter_result_path,
            _native_adapter_document(
                episode_index=episode_index,
                batch_result=batch_result,
                official_result=official_result,
                trace=trace,
                outcome=outcome,
                trajectory_source=trajectory_source,
            ),
        )
        private_context: dict[str, Any] = {
            "source": "frozen_benchmark_and_evaluator_gt_frame_observations",
            "nav_to_obj_candidate_total": len(candidate_names),
            "topdown_px_per_m": args.scene_px_per_m,
        }
        if target_context is not None:
            private_context["target"] = target_context
        if candidate_context:
            private_context["nav_to_obj_candidates"] = candidate_context
        _write_json(adapter_context_path, private_context)
        record.update(
            {
                "debug_dir": str(debug_dir),
                "target_name": target_name,
                "eligible_target_total": len(candidate_names),
                "eligible_target_observed_count": len(candidate_context),
                "target_selection": None if target_selection_path is None else str(target_selection_path),
                "sim_step_frame_manifest": str(frame_manifest) if frame_manifest.is_file() else None,
                "terminal_pose_h5": None if terminal_pose_path is None else str(terminal_pose_path),
                "trajectory_samples": len(trace),
                "trajectory_source": trajectory_source,
                "target_source": (
                    "evaluator_gt_frame_observation" if target_context is not None else "unavailable"
                ),
            }
        )
        try:
            render_episode_topdown(
                episode_result_path=adapter_result_path,
                benchmark_path=benchmark_file,
                debug_dir=debug_dir,
                output_path=image_path,
                private_context_path=adapter_context_path,
            )
        except Exception as error:
            record["status"] = "render_error"
            record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            _write_json(manifest_path, _manifest_payload(run_root, benchmark_file, records))
            if args.fail_fast:
                raise
            continue
        record["status"] = "rendered"
        records.append(record)
        _write_json(manifest_path, _manifest_payload(run_root, benchmark_file, records))
    payload = _manifest_payload(run_root, benchmark_file, records)
    _write_json(manifest_path, payload)
    return payload


def _manifest_payload(run_root: Path, benchmark_file: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root),
        "benchmark": str(benchmark_file),
        "rendered_count": sum(row.get("status") == "rendered" for row in records),
        "skipped_existing_count": sum(row.get("status") == "skipped_existing" for row in records),
        "error_count": sum(row.get("status") == "render_error" for row in records),
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Native batch manager run root")
    parser.add_argument("--benchmark", type=Path, required=True, help="NavToObj benchmark directory or JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNGs, adapters, and manifest")
    parser.add_argument(
        "--statuses",
        default="completed,failed",
        help="Comma-separated batch_result statuses to render (default: completed,failed)",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional global episode indices; default renders all currently terminal episodes.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap after episode-index sorting")
    parser.add_argument(
        "--scene-px-per-m",
        type=int,
        default=200,
        help="Static-map resolution for reports only (40-200; default: 200)",
    )
    parser.add_argument(
        "--manifest-name",
        default="topdown_batch_manifest.json",
        help="Output-local manifest name; use a unique name for concurrent shards",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing PNG/metadata pairs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first rendering error")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if not 40 <= args.scene_px_per_m <= 200:
        raise ValueError("--scene-px-per-m must be in [40, 200]")
    manifest_name = Path(args.manifest_name)
    if manifest_name.is_absolute() or ".." in manifest_name.parts or len(manifest_name.parts) != 1:
        raise ValueError("--manifest-name must be a simple filename")
    args.manifest_name = manifest_name.name
    payload = render_batch(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
