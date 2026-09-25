"""Recover timestamp-bounded public M2 context without replaying the simulator.

Old recorder boundaries contain asynchronously updated producer messages, not
atomic snapshots.  Each source is checked against the estimated request start
independently.  Exact-version cases and lagged reconstructions are separate
artifacts; neither is described as an exact original HTTP request.
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and _sys.path:
    _script_dir = _sys.path[0]
    _sys.path = [entry for entry in _sys.path if entry != _script_dir]

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


SCHEMA_VERSION = "interactive_nav_m2_context_source_v1"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    xy = [_number(item) for item in value[:2]]
    return xy if all(item is not None for item in xy) else None


def _public_copy(value: Any) -> Any:
    """Defense in depth after selecting only public producer messages."""
    if isinstance(value, Mapping):
        return {
            str(key): _public_copy(child)
            for key, child in value.items()
            if str(key).casefold() not in replay.FORBIDDEN_REQUEST_KEYS
        }
    if isinstance(value, list):
        return [_public_copy(child) for child in value]
    return value


def estimate_request_start(metric: Mapping[str, Any]) -> float:
    explicit = _number(metric.get("request_started_ts"))
    if explicit is not None:
        return explicit
    completed = _number(metric.get("timestamp"))
    latency = _number(metric.get("latency_s"))
    if completed is None or latency is None or latency < 0:
        raise replay.DatasetError("Cannot estimate request start without timestamp and latency_s")
    return completed - latency


def map_pose(boundary: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a map-frame pose with both pose and TF availability timestamps."""
    xy = _xy(boundary.get("pose"))
    pose_stamp = _number(boundary.get("stamp_sec"))
    source_frame = str(boundary.get("pose_frame_id") or "").lstrip("/")
    graph_frame = str(boundary.get("graph_frame_id") or "").lstrip("/")
    if xy is None or pose_stamp is None or not source_frame or not graph_frame:
        return None
    stamps = {"pose_timestamp": pose_stamp}
    if source_frame != graph_frame:
        transform = boundary.get("tf_map_from_odom") or {}
        if (
            str(transform.get("source_frame") or "").lstrip("/") != source_frame
            or str(transform.get("target_frame") or "").lstrip("/") != graph_frame
        ):
            return None
        tx, ty, yaw, transform_stamp = (
            _number(transform.get(key)) for key in ("x", "y", "yaw", "stamp_sec")
        )
        if None in (tx, ty, yaw, transform_stamp):
            return None
        cosine, sine = math.cos(yaw), math.sin(yaw)
        xy = [tx + cosine * xy[0] - sine * xy[1], ty + sine * xy[0] + cosine * xy[1]]
        stamps["transform_timestamp"] = transform_stamp
    return {
        "xy": xy,
        "frame_id": graph_frame,
        "source_frame_id": source_frame,
        "source_xy": _xy(boundary.get("pose")),
        "transform": dict(boundary.get("tf_map_from_odom") or {}) if source_frame != graph_frame else None,
        "step": int(boundary.get("step_index", 0)),
        "available_timestamp": max(stamps.values()),
        **stamps,
    }


def _room_id(graph: dict[str, Any], xy: list[float]) -> str | None:
    replay._install_runtime_import_paths()
    from semantic_decision_py_pkg.room_context import room_context_for_xy

    value = room_context_for_xy(graph, xy).get("room_id")
    if value in (None, ""):
        return None
    value = str(value)
    return value if value.startswith("room_") else "room_" + value


def _selection_event(selection: Mapping[str, Any], step: int) -> dict[str, Any] | None:
    decision_id = str(selection.get("decision_id") or "")
    stamp = _number(selection.get("selected_at"))
    if not decision_id or stamp is None or not selection.get("candidate_id"):
        return None
    metadata = selection.get("metadata") or {}
    return {
        "decision_id": decision_id,
        "candidate_id": str(selection["candidate_id"]),
        "behavior_type": str(selection.get("behavior_type") or ""),
        "target_id": str(selection.get("target_id") or ""),
        "target_name": str(selection.get("target_name") or ""),
        "group_id": str(selection.get("executed_group_id") or ""),
        "history_key": str(selection.get("executed_history_key") or ""),
        "target_room_id": metadata.get("target_room_id", metadata.get("room_id")),
        "goal_xy": _xy(selection.get("goal_xyyaw")),
        "selected_at": stamp,
        "step": step,
        "observation_step": step,
    }


def _feedback_event(feedback: Mapping[str, Any]) -> dict[str, Any] | None:
    stamp = _number(feedback.get("timestamp"))
    decision_id = str(feedback.get("decision_id") or "")
    if stamp is None or not decision_id:
        return None
    detail = feedback.get("detail") or {}
    result = {
        "decision_id": decision_id,
        "timestamp": stamp,
        "status": str(feedback.get("status") or "UNKNOWN"),
        "success": feedback.get("success"),
    }
    for key in ("reason", "error", "failure_reason", "termination_reason"):
        if isinstance(detail.get(key), (str, int, float, bool)):
            result[key] = detail[key]
    return result


def decision_history_at(
    selections: Iterable[dict[str, Any]],
    feedback: Iterable[dict[str, Any]],
    cutoff: float,
    observation_step: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """No feedback or decision after the request start is eligible."""
    selected: dict[str, dict[str, Any]] = {}
    for entry in selections:
        if entry["selected_at"] > cutoff:
            continue
        key = entry["decision_id"]
        if key not in selected or entry["selected_at"] < selected[key]["selected_at"]:
            selected[key] = dict(entry)
    known_feedback: dict[str, dict[str, Any]] = {}
    for entry in feedback:
        key = entry["decision_id"]
        if key not in selected or not selected[key]["selected_at"] <= entry["timestamp"] <= cutoff:
            continue
        if key not in known_feedback or entry["timestamp"] > known_feedback[key]["timestamp"]:
            known_feedback[key] = entry
    result = []
    for entry in sorted(selected.values(), key=lambda row: (row["selected_at"], row["decision_id"]))[-limit:]:
        entry["steps_ago"] = max(0, observation_step - entry["observation_step"])
        known = known_feedback.get(entry["decision_id"])
        entry["result"] = str(known["status"]) if known else "PENDING"
        if known:
            entry["feedback_timestamp"] = known["timestamp"]
            entry["success"] = known.get("success")
            for key in ("reason", "error", "failure_reason", "termination_reason"):
                if key in known:
                    entry[key] = known[key]
        result.append(entry)
    return result


def room_history_at(events: Iterable[dict[str, Any]], cutoff: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous: Any = object()
    for entry in sorted(events, key=lambda row: (row["available_timestamp"], row["entry_step"])):
        if entry["available_timestamp"] > cutoff:
            continue
        current = entry["room_id"]
        if current == previous:
            continue
        previous = current
        result.append(dict(entry))
    return result


def candidate_history_from_decisions(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recover count/result facts only; do not invent frontier-gain statistics."""
    result: dict[str, dict[str, Any]] = {}
    for entry in history:
        keys = {str(entry.get("history_key") or ""), str(entry.get("candidate_id") or "")}
        for key in keys - {""}:
            stats = result.setdefault(key, {"selection_count": 0, "history_source": "recorded_decisions"})
            stats["selection_count"] += 1
            stats["last_selected_step"] = entry["observation_step"]
            stats["last_result"] = entry["result"]
            if entry["result"] not in {"UNKNOWN", "PENDING", "STARTED", "RUNNING"}:
                stats["last_terminal_result"] = entry["result"]
    return result


def recorded_frontier_lengths(raw_candidates: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    observed: dict[str, float] = defaultdict(float)
    eligible: dict[str, float] = defaultdict(float)
    seen: set[str] = set()
    for candidate in raw_candidates:
        if str(candidate.get("behavior_type") or "").upper() != "EXPLORE":
            continue
        metadata = candidate.get("metadata") or {}
        key = str(metadata.get("cluster_id") or candidate.get("candidate_id") or "")
        if key in seen:
            continue
        seen.add(key)
        room = metadata.get("room_id", metadata.get("target_room_id"))
        if room in (None, ""):
            continue
        room = str(room) if str(room).startswith("room_") else "room_" + str(room)
        length = _number(metadata.get("frontier_length_m"))
        if length is None:
            count = _number(metadata.get("cell_count"))
            resolution = _number(metadata.get("map_resolution"))
            length = count * resolution if count is not None and resolution is not None else None
        if length is None or length < 0:
            continue
        observed[room] += length
        if metadata.get("hard_constraints_passed", True) and metadata.get("room_reachable", True):
            eligible[room] += length
    return dict(observed), dict(eligible)


def candidate_map_pose(state: Mapping[str, Any], poses: list[dict[str, Any]]) -> dict[str, Any] | None:
    semantic = state.get("semantic") or {}
    xy = _xy(semantic.get("robot_xy"))
    stamp = _number(semantic.get("timestamp"))
    source_frame = str(state.get("semantic_pose_frame_id") or "").lstrip("/")
    graph_frame = str(state.get("semantic_graph_frame_id") or "").lstrip("/")
    if xy is None or stamp is None or not source_frame or not graph_frame:
        return None
    transform = None
    if source_frame != graph_frame:
        transforms = [
            row["transform"] for row in poses
            if row.get("transform") and row["source_frame_id"] == source_frame
            and row["frame_id"] == graph_frame
            and float(row["transform_timestamp"]) <= min(stamp, float(state["cutoff"]))
        ]
        if not transforms:
            return None
        transform = max(transforms, key=lambda item: float(item["stamp_sec"]))
        cosine, sine = math.cos(float(transform["yaw"])), math.sin(float(transform["yaw"]))
        xy = [float(transform["x"]) + cosine * xy[0] - sine * xy[1],
              float(transform["y"]) + sine * xy[0] + cosine * xy[1]]
    return {
        "xy": xy, "source_xy": _xy(semantic.get("robot_xy")),
        "frame_id": graph_frame, "source_frame_id": source_frame,
        "step": int(state["semantic_step"]), "pose_timestamp": stamp,
        "available_timestamp": stamp, "transform": transform,
        "transform_timestamp": transform["stamp_sec"] if transform else None,
        "source": "semantic_candidates_robot_xy_transformed",
        "source_frame_provenance": "candidate_producer_odom_callback_and_recorded_pose_frame",
    }


def _choose_graph(state: dict[str, Any], graph: dict[str, Any], step: int) -> None:
    stamp = _number(graph.get("timestamp"))
    revision = int(graph.get("graph_revision", -1))
    wanted = int(state["source"]["graph_revision"])
    if (
        stamp is None or stamp > state["cutoff"] or revision > wanted
        or str(graph.get("episode_id") or "") != str(state["source"]["episode_id"])
    ):
        return
    rank = (revision == wanted, revision, stamp)
    if rank > state.get("graph_rank", (False, -1, -1.0)):
        state.update(graph=graph, graph_step=step, graph_rank=rank)


def _choose_semantic(state: dict[str, Any], semantic: dict[str, Any], step: int) -> None:
    stamp = _number(semantic.get("timestamp"))
    sequence = int(semantic.get("sequence", -1))
    revision = int(semantic.get("graph_revision", -1))
    source = state["source"]
    if (
        stamp is None or stamp > state["cutoff"]
        or sequence > int(source["candidate_sequence"])
        or revision > int(source["graph_revision"])
        or str(semantic.get("episode_id") or "") != str(source["episode_id"])
    ):
        return
    exact = sequence == int(source["candidate_sequence"]) and revision == int(source["graph_revision"])
    rank = (exact, sequence, stamp)
    if rank > state.get("semantic_rank", (False, -1, -1.0)):
        state.update(semantic=semantic, semantic_step=step, semantic_rank=rank)


def enrich_attempt(cases: list[dict[str, Any]], attempt_dir: Path) -> list[dict[str, Any]]:
    metrics = list(replay._jsonl_rows(attempt_dir / "mllm_metrics.jsonl"))
    metric_by_key = defaultdict(list)
    for row in metrics:
        if row.get("role") == "subgoal_selection":
            metric_by_key[replay._metric_request_key(row)].append(row)
    states = []
    for case in cases:
        source = case["source"]
        key = (str(source["episode_id"]), int(source["graph_revision"]), int(source["candidate_sequence"]))
        matches = metric_by_key.get(key, [])
        if not matches:
            raise replay.DatasetError(f"No original M2 metric for {case['case_id']}")
        metric = min(matches, key=lambda row: abs(float(row["timestamp"]) - float(source["recorded_timestamp"])))
        states.append({"case": case, "source": source, "metric": metric, "cutoff": estimate_request_start(metric)})

    poses: list[dict[str, Any]] = []
    rooms: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    feedback: dict[tuple[str, float], dict[str, Any]] = {}
    seen_pose: set[tuple[int, float]] = set()
    path = attempt_dir / "debug" / "raw" / "step_boundaries.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                boundary = json.loads(line)
            except json.JSONDecodeError as exc:
                raise replay.DatasetError(f"{path}:{line_number}: {exc}") from exc
            step = int(boundary.get("step_index", 0))
            # Never retain gt_observations, scene/asset data or evaluator results.
            graph = boundary.get("unified_graph") or {}
            semantic = boundary.get("semantic_candidates") or {}
            for state in states:
                _choose_graph(state, graph, step)
                _choose_semantic(state, semantic, step)
                if state.get("graph") is graph:
                    state["graph_frame_id"] = str(boundary.get("graph_frame_id") or "")
                if state.get("semantic") is semantic:
                    state["semantic_pose_frame_id"] = str(boundary.get("pose_frame_id") or "")
                    state["semantic_graph_frame_id"] = str(boundary.get("graph_frame_id") or "")
            pose = map_pose(boundary)
            if pose and (pose["step"], pose["available_timestamp"]) not in seen_pose:
                seen_pose.add((pose["step"], pose["available_timestamp"]))
                poses.append(pose)
                graph_stamp = _number(graph.get("timestamp"))
                if graph_stamp is not None:
                    rooms.append({
                        "room_id": _room_id(graph, pose["xy"]),
                        "entry_xy": pose["xy"],
                        "entry_step": step,
                        "available_timestamp": max(pose["available_timestamp"], graph_stamp),
                        "pose_timestamp": pose["pose_timestamp"],
                        "graph_timestamp": graph_stamp,
                        "graph_revision": int(graph.get("graph_revision", 0)),
                    })
            selection = _selection_event(boundary.get("semantic_selection") or {}, step)
            if selection and selection["decision_id"] not in selections:
                selections[selection["decision_id"]] = selection
            event = _feedback_event(boundary.get("semantic_behavior_feedback") or {})
            if event:
                feedback[(event["decision_id"], event["timestamp"])] = event

    records = []
    for state in states:
        case, cutoff = state["case"], state["cutoff"]
        graph = dict(state.get("graph") or {})
        if graph:
            graph["frame_id"] = state.get("graph_frame_id") or None
        semantic = state.get("semantic") or {}
        known_poses = [row for row in poses if row["available_timestamp"] <= cutoff]
        initial = min(known_poses, key=lambda row: (row["step"], row["available_timestamp"])) if known_poses else None
        current = candidate_map_pose(state, poses)
        if current is None and known_poses:
            current = dict(max(known_poses, key=lambda row: (row["pose_timestamp"], row["step"])), source="latest_prior_recorder_pose_approximation")
        observation_step = int(state.get("semantic_step", state["source"].get("recorder_step", 0)))
        full_history = decision_history_at(selections.values(), feedback.values(), cutoff, observation_step, limit=max(1, len(selections)))
        history = full_history[-30:]
        candidate_history = candidate_history_from_decisions(full_history)
        observed_frontiers, eligible_frontiers = recorded_frontier_lengths(semantic.get("candidates") or [])
        visits = room_history_at(rooms, cutoff)
        missing = []
        if not graph:
            missing.append("graph_before_request")
        if not semantic:
            missing.append("semantic_candidates_before_request")
        if initial is None:
            missing.append("initial_recorded_map_xy")
        if current is None:
            missing.append("current_recorded_map_xy")
        exact_graph = bool(state.get("graph_rank", (False,))[0])
        exact_semantic = bool(state.get("semantic_rank", (False,))[0])
        strict = bool(exact_graph and exact_semantic and current and initial)
        robot = {
            "frame_id": current["frame_id"] if current else None,
            "position_frame_id": current["frame_id"] if current else None,
            "units": "m",
            "robot_xy": current["xy"] if current else [],
            "current_xy": current["xy"] if current else None,
            "initial_xy": initial["xy"] if initial else None,
            "initial_position_source": "first_recorded_pose_not_verified_reset_spawn",
            "initial_position_step": initial["step"] if initial else None,
            "current_position_step": current["step"] if current else None,
            "observation_step": observation_step,
            "room_visit_history": visits,
            "entered_room_ids": list(dict.fromkeys(entry["room_id"] for entry in visits if entry["room_id"])),
            "decision_history": history,
            "group_history": [],
            "candidate_history": candidate_history,
            "room_frontier_lengths": eligible_frontiers,
            "observed_room_frontier_lengths": observed_frontiers,
            "frontier_length_source": "recorded_raw_candidate_pool_lower_bound",
            "frontier_length_complete": False,
            "exploration_context": _public_copy(semantic.get("exploration_context") or {}),
        }
        graph_revision = int(graph.get("graph_revision", -1))
        reconstruction = {
            "mode": "timestamp_bounded_public_reconstruction",
            "subset": "exact_version_aligned" if strict else "lagged_or_incomplete",
            "strict_version_aligned": strict,
            "original_http_request_exact": False,
            "request_cutoff": cutoff,
            "cutoff_source": "request_started_ts" if "request_started_ts" in state["metric"] else "metrics_timestamp_minus_latency_s",
            "cutoff_is_estimate": "request_started_ts" not in state["metric"],
            "graph_exact_revision": exact_graph,
            "candidate_exact_sequence_and_revision": exact_semantic,
            "desired_graph_revision": int(state["source"]["graph_revision"]),
            "actual_graph_revision": graph_revision,
            "graph_revision_lag": int(state["source"]["graph_revision"]) - graph_revision if graph else None,
            "graph_timestamp": graph.get("timestamp"),
            "graph_age_s": cutoff - float(graph["timestamp"]) if graph else None,
            "graph_source_step": state.get("graph_step"),
            "semantic_timestamp": semantic.get("timestamp"),
            "semantic_source_step": state.get("semantic_step"),
            "semantic_candidate_sequence": semantic.get("sequence"),
            "raw_candidate_count": len(semantic.get("candidates") or []),
            "raw_pool_complete": False,
            "raw_pool_completeness_note": "Public producer snapshot only; upstream discarded frontiers cannot be recovered.",
            "history_count": len(history),
            "history_limit": 30,
            "history_complete": False,
            "history_completeness_note": "Recorded decisions only; compact sampling can miss intermediate publications.",
            "room_visit_count": len(visits),
            "room_history_method": "per_snapshot_map_pose_room_aabb_containment_consecutive_dedup",
            "missing_fields": missing,
            "initial_pose_provenance": initial,
            "current_pose_provenance": current,
            "semantic_robot_xy_untransformed": semantic.get("robot_xy"),
            "candidate_history_reconstructed": True,
            "candidate_history_scope": "all_timestamp_bounded_recorded_decisions_counts_and_results_only",
            "frontier_length_scope": "recorded_raw_candidate_pool_lower_bound_not_full_clusters",
            "eligible_frontier_scope": "producer_hard_constraints_and_reachability_only_before_current_curator",
            "current_pose_source": current.get("source") if current else None,
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case["case_id"],
            "source": dict(state["source"], raw_path=str(path)),
            "raw_candidates": _public_copy(semantic.get("candidates") or []),
            "graph": _public_copy(graph),
            "target_context": _public_copy(semantic.get("target_context") or {}),
            "robot_context": robot,
            "legacy_request": case["request"],
            "reconstruction": reconstruction,
            "diagnostics": {
                "original_candidate_options": state["metric"].get("candidate_options") or [],
                "original_candidate_pool_count": state["metric"].get("candidate_pool_count"),
                "original_curated_candidate_count": state["metric"].get("curated_candidate_count"),
            },
        }
        for name in ("raw_candidates", "graph", "target_context", "robot_context"):
            forbidden = replay._nested_forbidden_paths(record[name], name)
            if forbidden:
                raise replay.DatasetError("Private field retained: " + ", ".join(forbidden))
        records.append(record)
    return records


def build_dataset(cases_path: Path, log_root: Path, output_dir: Path, attempt_filter: str = "") -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in replay._jsonl_rows(cases_path):
        attempt = str(case["source"]["attempt"])
        if not attempt_filter or attempt_filter in attempt:
            grouped[attempt].append(case)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_dir / "enriched_records.jsonl",
        "strict": output_dir / "exact_version_aligned.jsonl",
        "reconstructed": output_dir / "lagged_or_incomplete.jsonl",
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_cases": str(cases_path.resolve()),
        "log_root": str(log_root.resolve()),
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "attempt_count": len(grouped),
        "case_count": 0,
        "exact_version_aligned_count": 0,
        "exact_graph_count": 0,
        "exact_candidate_count": 0,
        "missing_fields": Counter(),
        "history_count_distribution": Counter(),
        "graph_revision_lag_distribution": Counter(),
        "limitations": [
            "Request start is estimated unless explicitly recorded.",
            "Exact-version-aligned is not exact original HTTP request reconstruction.",
            "Recorded selection history is not guaranteed exhaustive.",
            "First recorded pose is not independently verified reset spawn.",
            "Graph facts come only from the public observed graph, never GT or scene assets.",
        ],
    }
    ages: list[float] = []
    with paths["all"].open("w", encoding="utf-8") as all_stream, paths["strict"].open("w", encoding="utf-8") as strict_stream, paths["reconstructed"].open("w", encoding="utf-8") as other_stream:
        for attempt, cases in sorted(grouped.items()):
            resolved = (log_root / attempt).resolve()
            if not resolved.is_relative_to(log_root.resolve()):
                raise replay.DatasetError(f"Attempt escapes log root: {attempt}")
            records = enrich_attempt(cases, resolved)
            for record in records:
                reconstruction = record["reconstruction"]
                serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                all_stream.write(serialized)
                (strict_stream if reconstruction["strict_version_aligned"] else other_stream).write(serialized)
                summary["case_count"] += 1
                summary["exact_version_aligned_count"] += int(reconstruction["strict_version_aligned"])
                summary["exact_graph_count"] += int(reconstruction["graph_exact_revision"])
                summary["exact_candidate_count"] += int(reconstruction["candidate_exact_sequence_and_revision"])
                summary["missing_fields"].update(reconstruction["missing_fields"])
                summary["history_count_distribution"][str(reconstruction["history_count"])] += 1
                summary["graph_revision_lag_distribution"][str(reconstruction["graph_revision_lag"])] += 1
                if reconstruction["graph_age_s"] is not None:
                    ages.append(reconstruction["graph_age_s"])
            all_stream.flush()
            strict_stream.flush()
            other_stream.flush()
            print(json.dumps({"attempt": attempt, "cases": len(records), "completed_cases": summary["case_count"]}), flush=True)
    summary["graph_age_s"] = {"mean": statistics.mean(ages), "max": max(ages)} if ages else None
    summary["artifact_bytes"] = {key: path.stat().st_size for key, path in paths.items()}
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", default="", help="Optional substring filter for one-attempt smoke extraction")
    args = parser.parse_args()
    summary = build_dataset(args.cases, args.log_root, args.output_dir, args.attempt)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
