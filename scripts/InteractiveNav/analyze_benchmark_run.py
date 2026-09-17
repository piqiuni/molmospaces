#!/usr/bin/env python3
"""Offline, reproducible analysis for an InteractiveNav benchmark run.

The analyzer never launches ROS, MuJoCo, or a model server.  It consumes the
committed batch summary and per-episode artifacts, then writes JSON, CSV,
Markdown, and optional visual contact sheets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable


SCHEMA_VERSION = "interactive_nav_offline_analysis_v1"
DEFAULT_BASELINES = (
    Path(__file__).resolve().parent
    / "configs/evaluation/benchmark_analysis_baselines_20260917.json"
)

LOG_PATTERNS = {
    "action_timeout": re.compile(r"action timeout, using noop", re.I),
    "reset_clear_costmaps_timeout": re.compile(r"clear_costmaps.*(?:failed|timeout)", re.I),
    "make_plan_unreachable": re.compile(r"make_plan_unreachable|no path|NO_PATH", re.I),
    "semantic_no_progress": re.compile(r"(?:reason.{0,40}semantic_subgoal_no_progress|semantic_subgoal_no_progress.{0,40}(?:failed|triggered))", re.I),
    "candidate_exhaustion": re.compile(r"(?:reason.{0,40}no_eligible_candidates|no eligible candidates after)", re.I),
    "capability_unavailable": re.compile(r"capability_unavailable", re.I),
    "target_claim_unverified": re.compile(r"target_claim_unverified", re.I),
    "mllm_timeout": re.compile(r"(?:attribute inference|mllm|model request).{0,80}(?:timed out|timeout exceeded)", re.I),
    "http_5xx": re.compile(r"HTTP/\d(?:\.\d)?\s+5\d\d|502 Bad Gateway", re.I),
    "tf_wait_warning": re.compile(r"TF.{0,80}(?:timeout|failed)|Extrapolation Error", re.I),
    "image_stamp_mismatch": re.compile(r"image_stamp_mismatch", re.I),
    "make_plan_preflight_unavailable": re.compile(
        r"make_plan preflight unavailable.{0,160}(?:b''|empty)", re.I
    ),
    "actionlib_wait_without_goal": re.compile(r"wait_for_result when no goal exists", re.I),
    "actionlib_preempt_done_race": re.compile(r"PREEMPTING.*simple state DONE", re.I),
    "rear_goal_recovery_refused": re.compile(r"rear-goal safe recovery refused", re.I),
    "rear_goal_heading_unavailable": re.compile(r"rear_goal_heading_unavailable", re.I),
    "navigation_stagnation": re.compile(r"navigation_stagnation", re.I),
    "oom": re.compile(r"out of memory|OutOfMemory", re.I),
    "segfault": re.compile(r"segmentation fault|SIGSEGV", re.I),
    "recorder_drop": re.compile(r"(?:dropped_frames?[=: ]+[1-9]|queue overflow.{0,40}drop|write failure)", re.I),
}


# Reuse the established round loader/metric contract instead of reimplementing
# attempt selection and paper-metric denominators here.  The fallback keeps
# small synthetic fixtures usable when this file is loaded directly in tests.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
try:
    from evaluation.v3_round_summary import summarise_round
except ImportError:  # pragma: no cover - only relevant outside the repository
    summarise_round = None


TARGET_LABEL_ALIASES = {
    "crapper": "toilet",
    "toilet": "toilet",
    "ashcan": "garbage can",
    "garbagecan": "garbage can",
    "atomizer": "spray bottle",
    "spraybottle": "spray bottle",
    "alarmclock": "alarm clock",
    "irishpotato": "potato",
    "compactdisk": "cd",
    "television": "television",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a stable provenance record without loading large artifacts at once."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "present": False}
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "present": True,
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def group_for_index(index: int) -> str:
    if index < 1000:
        return "channel"
    if index < 2000:
        return "container"
    return "mixed"


def numeric(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else None


def canonical_label(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return TARGET_LABEL_ALIASES.get(text, text)


def load_round_contract(evaluation_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    if summarise_round is None:
        return None, "v3_round_summary import unavailable"
    try:
        return summarise_round(evaluation_dir), None
    except Exception as exc:  # synthetic/legacy fixtures may be intentionally sparse
        return None, f"v3_round_summary failed: {type(exc).__name__}: {exc}"


def _mask_pixels(observation: dict[str, Any]) -> int | None:
    counts = ((observation.get("mask_rle") or {}).get("counts"))
    if isinstance(counts, list):
        try:
            return int(sum(int(value) for value in counts[1::2]))
        except (TypeError, ValueError):
            return None
    bbox = observation.get("bbox_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return max(0, int(bbox[2]) - int(bbox[0]) + 1) * max(
                0, int(bbox[3]) - int(bbox[1]) + 1
            )
        except (TypeError, ValueError):
            return None
    return None


def analyze_frame_manifest(
    manifest_path: Path,
    *,
    target_xy: list[float] | tuple[float, float] | None,
    target_object_name: str | None,
    relaxed_instance_id: str | None,
) -> dict[str, Any]:
    """Stream public per-frame GT metadata without loading images or the file at once."""
    if not manifest_path.is_file():
        return {
            "present": False,
            "path": str(manifest_path),
            "frame_count": 0,
            "runtime_start_stamp": None,
            "runtime_end_stamp": None,
            "longest_stationary_pose_run": None,
            "strict_target_instance": None,
            "relaxed_target_instance": None,
            "relaxed_to_strict_target_distance_m": None,
            "same_category_instances": [],
        }

    opener = gzip.open if manifest_path.suffix == ".gz" else open
    objects: dict[str, dict[str, Any]] = {}
    frame_count = 0
    runtime_start = None
    runtime_end = None
    previous_pose = None
    stationary_start = None
    stationary_length = 0
    longest_stationary = {"start_step": None, "end_step": None, "frame_count": 0}

    with opener(manifest_path, "rt", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            frame_count += 1
            step = int(row.get("step_index") or 0)
            stamp = numeric(row.get("stamp_sec"))
            if stamp is not None:
                runtime_start = stamp if runtime_start is None else min(runtime_start, stamp)
                runtime_end = stamp if runtime_end is None else max(runtime_end, stamp)
            gt = row.get("gt_observations") or {}
            pose = gt.get("observation_pose_xyyaw")
            if isinstance(pose, list) and len(pose) >= 3:
                pose = [float(pose[0]), float(pose[1]), float(pose[2])]
                if previous_pose is not None:
                    translation = math.hypot(pose[0] - previous_pose[0], pose[1] - previous_pose[1])
                    yaw_delta = abs(math.atan2(math.sin(pose[2] - previous_pose[2]), math.cos(pose[2] - previous_pose[2])))
                    if translation <= 0.002 and yaw_delta <= 0.002:
                        if stationary_length == 0:
                            stationary_start = step - 1
                            stationary_length = 2
                        else:
                            stationary_length += 1
                    else:
                        stationary_length = 0
                        stationary_start = None
                    if stationary_length > longest_stationary["frame_count"]:
                        longest_stationary = {
                            "start_step": stationary_start,
                            "end_step": step,
                            "frame_count": stationary_length,
                        }
                previous_pose = pose

            for observation in gt.get("observations") or []:
                object_id = str(observation.get("id") or "")
                if not object_id:
                    continue
                item = objects.setdefault(
                    object_id,
                    {
                        "instance_id": object_id,
                        "name": observation.get("name"),
                        "canonical_name": canonical_label(observation.get("name")),
                        "center_xy": None,
                        "visible_frame_count": 0,
                        "first_visible_step": step,
                        "last_visible_step": step,
                        "max_mask_pixels": 0,
                        "max_mask_step": step,
                    },
                )
                item["visible_frame_count"] += 1
                item["last_visible_step"] = step
                center = ((observation.get("box_3d") or {}).get("center"))
                if isinstance(center, list) and len(center) >= 2:
                    item["center_xy"] = [float(center[0]), float(center[1])]
                pixels = _mask_pixels(observation)
                if pixels is not None and pixels > item["max_mask_pixels"]:
                    item["max_mask_pixels"] = pixels
                    item["max_mask_step"] = step

    target_label = canonical_label((target_object_name or "").split("_")[0])
    category_objects = [
        item for item in objects.values()
        if target_label and item.get("canonical_name") == target_label
    ]
    for item in category_objects:
        center = item.get("center_xy")
        item["distance_to_strict_target_m"] = (
            math.hypot(center[0] - float(target_xy[0]), center[1] - float(target_xy[1]))
            if center and target_xy and len(target_xy) >= 2
            else None
        )
    category_objects.sort(
        key=lambda item: (
            math.inf if item.get("distance_to_strict_target_m") is None else item["distance_to_strict_target_m"],
            item["instance_id"],
        )
    )
    strict = category_objects[0] if category_objects else None
    relaxed = objects.get(str(relaxed_instance_id)) if relaxed_instance_id else None
    relaxed_distance = None
    if relaxed and relaxed.get("center_xy") and target_xy and len(target_xy) >= 2:
        center = relaxed["center_xy"]
        relaxed_distance = math.hypot(center[0] - float(target_xy[0]), center[1] - float(target_xy[1]))
        relaxed = dict(relaxed, distance_to_strict_target_m=relaxed_distance)
    return {
        "present": True,
        "path": str(manifest_path),
        "frame_count": frame_count,
        "runtime_start_stamp": runtime_start,
        "runtime_end_stamp": runtime_end,
        "longest_stationary_pose_run": longest_stationary,
        "strict_target_instance": strict,
        "relaxed_target_instance": relaxed,
        "relaxed_to_strict_target_distance_m": relaxed_distance,
        "same_category_instances": category_objects,
    }


def collect_graph_revision_summary(path: Path, target_object_name: str | None) -> dict[str, Any]:
    """Summarize the persisted semantic-graph history without replaying ROS."""
    if not path.is_file():
        return {"present": False, "path": str(path)}
    event_counts = Counter()
    labels = Counter()
    relations = Counter()
    state_transitions = Counter()
    target_nodes: set[str] = set()
    revisions: list[int] = []
    steps: list[int] = []
    invalid_rows = 0
    target_label = canonical_label((target_object_name or "").split("_")[0])
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_rows += 1
                    continue
                event = str(row.get("event") or "UNKNOWN")
                event_counts[event] += 1
                if row.get("graph_revision") is not None:
                    revisions.append(int(row["graph_revision"]))
                if row.get("step_id") is not None:
                    steps.append(int(row["step_id"]))
                label = row.get("label")
                if label:
                    labels[str(label)] += 1
                    if target_label and canonical_label(str(label)) == target_label and row.get("node_id"):
                        target_nodes.add(str(row["node_id"]))
                if row.get("relation"):
                    relations[str(row["relation"])] += 1
                if event == "STATE_CHANGED":
                    state_transitions[f"{row.get('before')}->{row.get('after')}"] += 1
    except OSError:
        return {"present": False, "path": str(path), "read_error": True}
    return {
        "present": True,
        "path": str(path),
        "event_count": sum(event_counts.values()),
        "event_counts": dict(event_counts),
        "last_graph_revision": max(revisions) if revisions else None,
        "first_step_id": min(steps) if steps else None,
        "last_step_id": max(steps) if steps else None,
        "node_label_counts": dict(labels),
        "edge_relation_counts": dict(relations),
        "state_transition_counts": dict(state_transitions),
        "target_category_node_ids": sorted(target_nodes),
        "invalid_row_count": invalid_rows,
    }


def collect_log_evidence(
    attempt_dir: Path,
    runtime_window: tuple[float | None, float | None] | None = None,
) -> tuple[dict[str, int], dict[str, list[str]], list[str]]:
    candidates = [
        attempt_dir / "eval.log",
        attempt_dir / "runner.log",
        attempt_dir / "roslaunch.log",
        attempt_dir / "agent_launch.log",
        attempt_dir / "recorder.log",
        attempt_dir / "offline_video.log",
        attempt_dir / "debug/move_base_rosout.log",
    ]
    ros_log_root = attempt_dir / "ros_home/log"
    for pattern in (
        "*/semantic_behavior_executor-*.log",
        "*/semantic_rule_decision_node-*.log",
        "*/semantic_candidate_node-*.log",
        "*/semantic_mapping_py-*.log",
        "*/interaction_attribute_inference-*.log",
        "*/explore_py-*.log",
    ):
        candidates.extend(ros_log_root.glob(pattern))
    counts = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    scanned = []
    seen_files: set[tuple[int, int]] = set()
    runtime_start, runtime_end = runtime_window or (None, None)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_files:
            continue
        seen_files.add(identity)
        scanned.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_index, line in enumerate(lines):
            # roslaunch dumps the parameter server and the runner prints its
            # effective config.  Those lines contain words such as timeout and
            # drop_oldest, but are not runtime events.
            stripped = line.strip()
            is_config_line = stripped.startswith("* /") or stripped.startswith("[v3-eval]")
            timestamp_match = re.match(r"\[([0-9]+(?:\.[0-9]+)?)\]", stripped)
            line_timestamp = float(timestamp_match.group(1)) if timestamp_match else None
            outside_runtime = bool(
                line_timestamp is not None
                and (
                    (runtime_start is not None and line_timestamp < runtime_start - 5.0)
                    or (runtime_end is not None and line_timestamp > runtime_end + 5.0)
                )
            )
            if "Traceback (most recent call last)" in line:
                before = "\n".join(lines[max(0, line_index - 12):line_index])
                after = "\n".join(lines[line_index:line_index + 30])
                shutdown = (
                    "killing on exit" in before
                    and (
                        "publish() to a closed topic" in after
                        or "ROSInterruptException" in after
                        or "ROS shutdown request" in after
                    )
                )
                name = "shutdown_traceback" if shutdown else "runtime_traceback"
                counts[name] += 1
                if len(samples[name]) < 3:
                    last = after.splitlines()[-1] if after else line
                    samples[name].append(f"{path.name}: {line[:240]} ... {last[:240]}")
            for name, pattern in LOG_PATTERNS.items():
                # move_base emits a large startup/teardown TF storm.  Only
                # runtime-window TF evidence is relevant to episode causality.
                if name == "tf_wait_warning" and outside_runtime:
                    continue
                if not is_config_line and pattern.search(line):
                    counts[name] += 1
                    if len(samples[name]) < 3:
                        samples[name].append(f"{path.name}: {line[:500]}")
    return dict(counts), dict(samples), scanned


def collect_mllm_metrics(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    latencies = [value for row in rows if (value := numeric(row.get("latency_s"))) is not None]
    queue_lags = [value for row in rows if (value := numeric(row.get("queue_lag_sec"))) is not None]
    errors = [row for row in rows if str(row.get("error") or "").strip()]
    return {
        "path": str(path),
        "present": path.is_file(),
        "call_count": len(rows),
        "role_counts": dict(Counter(str(row.get("role") or "unknown") for row in rows)),
        "error_count": len(errors),
        "error_counts": dict(Counter(str(row.get("error")) for row in errors)),
        "errors": [
            {
                "role": row.get("role"),
                "error": row.get("error"),
                "observation_capture_step": row.get("observation_capture_step"),
                "queue_lag_sec": numeric(row.get("queue_lag_sec")),
                "timeout_s": numeric(row.get("timeout_s")),
                "latency_s": numeric(row.get("latency_s")),
            }
            for row in errors
        ],
        "latency_seconds": {
            "mean": mean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "queue_lag_seconds": {
            "mean": mean(queue_lags),
            "p50": percentile(queue_lags, 0.50),
            "p95": percentile(queue_lags, 0.95),
            "p99": percentile(queue_lags, 0.99),
            "max": max(queue_lags) if queue_lags else None,
        },
        "_latencies": latencies,
        "_queue_lags": queue_lags,
    }


def interaction_summary(
    attempts: list[dict[str, Any]], result: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = result or {}
    failure_reasons = Counter()
    status_counts = Counter()
    node_type_counts = Counter()
    successful = 0
    state_contract_warnings = []
    compact_attempts = []
    for item in attempts:
        successful += bool(item.get("success"))
        status_counts[str(item.get("status") or item.get("result_status") or "unknown")] += 1
        node_type_counts[str(item.get("node_type") or "unknown")] += 1
        if not item.get("success"):
            failure_reasons[str(item.get("failure_reason") or item.get("reason") or "unknown")] += 1
        post_state = item.get("post_state")
        if item.get("success") and post_state in (None, "closed", "unknown", "unavailable"):
            state_contract_warnings.append({
                "decision_step": item.get("decision_step"),
                "instance_id": item.get("instance_id") or item.get("object_id"),
                "node_type": item.get("node_type"),
                "post_state": post_state,
                "verification_source": item.get("verification_source"),
            })
        compact_attempts.append({
            "decision_step": item.get("decision_step"),
            "decision_id": item.get("decision_id"),
            "candidate_id": item.get("candidate_id"),
            "instance_id": item.get("instance_id") or item.get("object_id"),
            "node_type": item.get("node_type"),
            "action": item.get("action") or item.get("operation"),
            "success": bool(item.get("success")),
            "status": item.get("status") or item.get("result_status"),
            "failure_reason": item.get("failure_reason") or item.get("reason"),
            "pre_state": item.get("pre_state"),
            "post_state": post_state,
            "verification_source": item.get("verification_source"),
            "retryable": item.get("retryable"),
        })
    return {
        "attempt_count": len(attempts),
        "success_count": successful,
        "failure_count": len(attempts) - successful,
        "status_counts": dict(status_counts),
        "node_type_counts": dict(node_type_counts),
        "failure_reasons": dict(failure_reasons),
        "correct_count": int(result.get("correct_interaction_action_count") or 0),
        "extra_count": int(result.get("extra_interaction_action_count") or 0),
        "invalid_count": int(result.get("invalid_interaction_action_count") or 0),
        "error_count": int(result.get("error_interaction_attempt_count") or 0),
        "failed_count": int(result.get("failed_interaction_attempt_count") or 0),
        "irrelevant_count": int(result.get("task_irrelevant_interaction_attempt_count") or 0),
        "repeated_count": int(result.get("repeated_interaction_attempt_count") or 0),
        "state_contract_warning_count": len(state_contract_warnings),
        "state_contract_warnings": state_contract_warnings,
        "attempts": compact_attempts,
    }


def trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter()
    timings = []
    slowest: dict[str, Any] | None = None
    for event in trace:
        action = event.get("action")
        if isinstance(action, dict):
            action_counts[str(action.get("kind") or "unknown")] += 1
        timing = event.get("timing_ms")
        total_ms = numeric(timing.get("total")) if isinstance(timing, dict) else None
        if total_ms is None:
            continue
        seconds = total_ms / 1000.0
        timings.append(seconds)
        if slowest is None or seconds > slowest["seconds"]:
            slowest = {
                "decision_step": event.get("decision_step"),
                "seconds": seconds,
                "action_kind": action.get("kind") if isinstance(action, dict) else None,
            }
    return {
        "event_count": len(trace),
        "timed_event_count": len(timings),
        "action_counts": dict(action_counts),
        "mean_step_seconds": mean(timings),
        "p50_step_seconds": percentile(timings, 0.50),
        "p95_step_seconds": percentile(timings, 0.95),
        "p99_step_seconds": percentile(timings, 0.99),
        "slowest_event": slowest,
        "_timings": timings,
    }


def nearest_interaction_error(topdown: dict[str, Any]) -> float | None:
    gt = [item.get("xy") for item in topdown.get("gt_interactions", []) if item.get("xy")]
    actual = [item.get("xy") for item in topdown.get("actual_interactions", []) if item.get("xy")]
    if not gt or not actual:
        return None
    distances = []
    for ax, ay in actual:
        for gx, gy in gt:
            distances.append(math.hypot(float(ax) - float(gx), float(ay) - float(gy)))
    return min(distances) if distances else None


def classify_episode(result: dict[str, Any], logs: dict[str, int]) -> tuple[str, list[str], list[str]]:
    reason = str(result.get("terminal_reason") or "unknown")
    interactions = result.get("interaction_attempts") or []
    failures = Counter(
        str(item.get("failure_reason") or item.get("reason") or "unknown")
        for item in interactions
        if not item.get("success")
    )
    evidence = []
    recommendations = []

    if reason == "target_found":
        if result.get("success"):
            primary = "verified_success"
            evidence.append("Public goal evidence and formal task conditions both passed.")
        elif result.get("task_success"):
            primary = "formal_success_blocked_by_interaction_requirement"
            evidence.append("Target was found, but the required-interaction condition did not pass.")
            recommendations.append("Audit whether the frozen interaction requirement is necessary and whether the executed route crossed the intended interaction edge.")
        else:
            primary = "target_found_metric_inconsistency"
            evidence.append("Terminal reason says target_found while task_success is false.")
            recommendations.append("Add a metric-contract assertion for target_found versus task_success.")
    elif reason == "target_claim_unverified":
        if result.get("goal_definition_relaxed_success"):
            primary = "wrong_instance_same_category_target_claim"
            evidence.append(
                "The frozen target-instance claim failed, while the diagnostic category-level verifier accepted another same-category instance."
            )
        elif (numeric(result.get("target_distance_m")) or math.inf) <= 1.5:
            primary = "near_target_without_public_visual_evidence"
            evidence.append("Robot reached the success-radius neighborhood without a verifiable target observation.")
        else:
            primary = "ungrounded_or_premature_target_claim"
            evidence.append("Goal claim was emitted while distance/visibility evidence remained insufficient.")
        recommendations.append("Gate SUCCEEDED claims on a fresh restricted-frame observation ID and persist the rejected claim evidence for diagnosis.")
        if result.get("goal_definition_relaxed_success"):
            recommendations.append("Preserve instance identity and containing-object relationships through graph merges; category agreement alone must not end an instance-goal episode.")
    elif reason == "ros_bridge_observation_turn_limit":
        if failures.get("capability_unavailable"):
            primary = "budget_exhaustion_after_wrong_interaction_binding"
            evidence.append("At least one selected interaction object was reported capability_unavailable.")
            recommendations.append("Filter portal/container candidates by observed capability before committing an interaction request.")
        else:
            primary = "observation_budget_exhausted_without_goal"
            evidence.append("Observation-turn guard reached the episode budget before verified completion.")
        no_fresh = int(result.get("no_fresh_action_count") or 0)
        steps = max(1, int(result.get("step_count") or 0))
        if no_fresh / steps >= 0.10:
            evidence.append(f"No-fresh-action ratio was {no_fresh / steps:.1%}.")
            recommendations.append("Expose no-fresh-action recovery as a first-class decision outcome and stop repeated no-op turns earlier.")
        recommendations.append("Use remaining-budget-aware exploration so the last turns prioritize unresolved target/interaction evidence.")
    elif reason == "policy_exploration_stalled":
        termination = result.get("policy_termination") or {}
        terminal_detail = termination.get("terminal_detail") or {}
        latest_detail = ((termination.get("latest_goal_status") or {}).get("detail") or {})
        reported = str(
            latest_detail.get("reason")
            or terminal_detail.get("reported_reason")
            or "exploration_stalled"
        )
        if reported == "no_eligible_candidates_after_bounded_recovery":
            context = latest_detail.get("exploration_context") or {}
            raw_clusters = int(context.get("raw_frontier_cluster_count") or 0)
            raw_cells = int(context.get("raw_frontier_cell_count") or 0)
            proposals = int(context.get("proposal_count") or 0)
            if raw_clusters or raw_cells:
                primary = "candidate_generation_exhausted_despite_raw_frontiers"
                evidence.append(
                    f"Bounded recovery ended with {raw_clusters} raw frontier clusters / "
                    f"{raw_cells} cells but only {proposals} proposals."
                )
                recommendations.append(
                    "Trace frontier-to-proposal rejection reasons and retain one deterministic safe fallback when physical frontiers remain."
                )
            else:
                primary = "candidate_generation_exhausted_after_recovery"
                evidence.append("Bounded recovery ended without an executable candidate.")
                recommendations.append(
                    "Persist the rejected candidate set and bounded-recovery state for deterministic replay."
                )
        elif latest_detail.get("interaction_approach_options_exhausted"):
            mission_elapsed = int(latest_detail.get("mission_elapsed_task_steps") or 0)
            mission_limit = int(latest_detail.get("mission_timeout_task_steps") or 0)
            subgoal_elapsed = int(latest_detail.get("subgoal_elapsed_task_steps") or 0)
            if subgoal_elapsed == 0 and mission_limit and mission_elapsed >= mission_limit:
                primary = "global_mission_timer_triggered_on_new_subgoal"
                evidence.append(
                    f"Global no-progress timer was {mission_elapsed}/{mission_limit} steps while the newly selected subgoal was at step 0."
                )
                recommendations.append(
                    "Audit which verified progress events reset the global mission timer; do not terminate a fresh executable subgoal solely on stale mission age."
                )
            else:
                primary = "portal_approach_exhausted_after_navigation_stagnation"
                evidence.append(
                    f"Interaction approach options were exhausted after {subgoal_elapsed} subgoal steps "
                    f"(mission {mission_elapsed}/{mission_limit or 'N/A'})."
                )
                recommendations.append(
                    "Blacklist failed portal approach anchors, retain the failure geometry, and select a materially different reachable anchor."
                )
        else:
            primary = "semantic_mission_no_progress"
            evidence.append(f"Policy reported {reported}.")
            recommendations.append(
                "Persist the full goal-status detail and executable candidate set when semantic mission progress expires."
            )
    elif reason == "cross_subgoal_navigation_stall":
        action_lifecycle = (
            logs.get("actionlib_wait_without_goal", 0)
            + logs.get("rear_goal_recovery_refused", 0)
            + logs.get("rear_goal_heading_unavailable", 0)
        )
        primary = (
            "tf_action_lifecycle_stall"
            if action_lifecycle and logs.get("tf_wait_warning", 0)
            else "repeated_navigation_failures_with_low_displacement"
        )
        early = result.get("early_stop") or {}
        evidence.append(
            f"Early-stop saw {early.get('failed_subgoal_count', 0)} failed subgoals with "
            f"{float(early.get('displacement_m') or 0):.3f} m displacement."
        )
        if primary == "tf_action_lifecycle_stall":
            evidence.append(
                f"Runtime logs contain {logs.get('tf_wait_warning', 0)} TF warnings and "
                f"{action_lifecycle} action/rear-goal lifecycle failures."
            )
            recommendations.append(
                "Repair the TF timestamp/action-goal lifecycle before tuning high-level exploration; reject stale poses and never wait on a missing action goal."
            )
        else:
            recommendations.append("Deduplicate equivalent failed subgoals and blacklist locally unreachable goals before selecting the next frontier.")
    else:
        primary = f"unclassified_{reason}"
        evidence.append(f"No specialized offline rule exists for terminal reason {reason}.")
        recommendations.append("Add an explicit analyzer rule and a synthetic fixture before using this terminal reason in reports.")

    if logs.get("make_plan_unreachable"):
        evidence.append(f"Logs contain {logs['make_plan_unreachable']} unreachable-plan signals.")
    if logs.get("action_timeout"):
        evidence.append(f"Logs contain {logs['action_timeout']} bridge action timeouts/noops.")
    if logs.get("runtime_traceback") or logs.get("oom") or logs.get("segfault"):
        evidence.append("Infrastructure failure signature was detected in logs.")
        recommendations.append("Treat infrastructure exceptions as incomplete runs, never as navigation failures.")
    return primary, evidence, list(dict.fromkeys(recommendations))


def add_contextual_findings(
    result: dict[str, Any],
    spatial: dict[str, Any],
    interaction: dict[str, Any],
    evidence: list[str],
    recommendations: list[str],
) -> None:
    failures = interaction.get("failure_reasons") or {}
    if failures:
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(failures.items()))
        evidence.append(f"Interaction failures: {rendered}.")
    if failures.get("interaction_wrong_face"):
        recommendations.append("Validate the selected articulation face against the current observation before executing the interaction macro.")
    if failures.get("interaction_not_visible"):
        recommendations.append("Require a fresh visible interaction observation before dispatch; otherwise re-observe instead of consuming an interaction attempt.")
    if failures.get("drawer_scan_execution_failed"):
        recommendations.append("Record the failed drawer-scan stage and allow a bounded re-approach before abandoning the container sequence.")
    if failures.get("capability_unavailable"):
        recommendations.append("Filter candidates whose observed interaction capability is unavailable before request dispatch.")

    interaction_error = spatial.get("nearest_actual_to_gt_interaction_m")
    if interaction_error is not None and interaction_error > 1.0:
        evidence.append(f"Nearest actual interaction was {interaction_error:.2f} m from the required GT interaction.")
        recommendations.append("Use topology/required-edge context to disambiguate nearby but task-irrelevant interaction objects.")

    coverage = spatial.get("coverage_ratio")
    distance = numeric(result.get("target_distance_m"))
    visibility = numeric(result.get("target_visibility_fraction"))
    if coverage is not None and coverage < 0.30 and not result.get("task_success"):
        evidence.append(f"Map coverage was low ({coverage:.1%}).")
        recommendations.append("Prioritize coverage recovery when the mapped-free area remains below the episode-specific exploration floor.")
    if coverage is not None and coverage >= 0.85 and not result.get("task_success"):
        evidence.append(f"Map coverage was already high ({coverage:.1%}); failure is not explained by insufficient global mapping alone.")
    if distance is not None and distance <= 1.75 and (visibility is None or visibility <= 0.0001) and not result.get("task_success"):
        evidence.append(f"Robot ended near the target ({distance:.2f} m) without robust visual evidence.")
        recommendations.append("Add a short terminal view-search at near-target poses before declaring exhaustion or emitting success.")

    # Keep recommendations stable and non-repeating after contextual additions.
    recommendations[:] = list(dict.fromkeys(recommendations))


def image_metrics(paths: list[Path]) -> dict[str, Any]:
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError:
        return {"available": False, "error": "Pillow is not installed"}
    images = []
    luminance = []
    dark_fraction = []
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        images.append(image)
        gray = image.convert("L").resize((160, 120))
        histogram = gray.histogram()
        pixels = sum(histogram)
        luminance.append(sum(index * count for index, count in enumerate(histogram)) / pixels)
        dark_fraction.append(sum(histogram[:32]) / pixels)
    change = None
    if len(images) >= 2:
        first = images[0].resize((160, 120))
        last = images[-1].resize((160, 120))
        change = statistics.fmean(ImageStat.Stat(ImageChops.difference(first, last)).mean)
    return {
        "available": bool(images),
        "sample_count": len(images),
        "mean_luminance": mean(luminance),
        "mean_dark_fraction": mean(dark_fraction),
        "first_last_mean_abs_difference": change,
    }


def frame_samples(attempt_dir: Path, offline: dict[str, Any]) -> list[Path]:
    count = int(offline.get("sim_frame_count") or offline.get("output_frame_count") or 0)
    if count <= 0:
        return []
    frame_dir = attempt_dir / "sim_step_frames"
    selected = [0, (count - 1) // 2, count - 1]
    result = []
    for index in selected:
        path = frame_dir / f"step_{index:06d}.png"
        if path.is_file():
            result.append(path)
    return result


def event_frame_samples(
    attempt_dir: Path,
    interactions: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    frame_dir = attempt_dir / "sim_step_frames"
    requested: list[tuple[str, int]] = []
    for item in interactions.get("attempts") or []:
        step = item.get("decision_step")
        if step is None:
            continue
        for delta, suffix in ((-10, "before"), (0, "at"), (10, "after")):
            requested.append((f"interaction_{suffix}:{item.get('instance_id')}", max(0, int(step) + delta)))
    relaxed = manifest.get("relaxed_target_instance") or {}
    if relaxed.get("max_mask_step") is not None:
        requested.append((f"relaxed_claim:{relaxed.get('instance_id')}", int(relaxed["max_mask_step"])))
    result = []
    seen = set()
    for kind, step in requested:
        path = frame_dir / f"step_{step:06d}.png"
        if path.is_file() and (kind, step) not in seen:
            seen.add((kind, step))
            result.append({"kind": kind, "step": step, "path": str(path)})
    return result


def analyze_episode(evaluation_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    index = int(row["episode_index"])
    episode_dir = evaluation_dir / f"episode_{index:04d}"
    attempt_dir = Path(row.get("attempt_dir") or episode_dir / str(row.get("attempt") or "attempt_001"))
    result_path = Path(str(row.get("episode_result_path") or ""))
    if not result_path.is_file():
        matches = list(attempt_dir.glob("eval/episodes/*/episode_result.json"))
        if len(matches) != 1:
            raise ValueError(f"Episode {index}: expected one result, found {len(matches)}")
        result_path = matches[0]
    document = read_json(result_path, {})
    result = document.get("result") or {}
    trace = document.get("trace") or []
    topdown_path = episode_dir / "episode_topdown.json"
    topdown = read_json(topdown_path, {})
    debug_summary_path = attempt_dir / "debug/summary.json"
    debug_summary = read_json(debug_summary_path, {})
    offline_path = attempt_dir / "offline_video_summary.json"
    offline = read_json(offline_path, {})
    target_payload = topdown.get("gt_target") or {}
    target_xy = target_payload.get("xy")
    target_object_name = target_payload.get("object_name")
    manifest_path = attempt_dir / "sim_step_frames/manifest.jsonl"
    if not manifest_path.is_file() and manifest_path.with_suffix(".jsonl.gz").is_file():
        manifest_path = manifest_path.with_suffix(".jsonl.gz")
    manifest = analyze_frame_manifest(
        manifest_path,
        target_xy=target_xy,
        target_object_name=target_object_name,
        relaxed_instance_id=result.get("goal_definition_relaxed_instance_id"),
    )
    graph_revision_path = attempt_dir / "debug/graph/graph_revision_events.jsonl"
    graph_revision = collect_graph_revision_summary(
        graph_revision_path, target_object_name
    )
    log_counts, log_samples, scanned_logs = collect_log_evidence(
        attempt_dir,
        (manifest.get("runtime_start_stamp"), manifest.get("runtime_end_stamp")),
    )
    mllm = collect_mllm_metrics(attempt_dir / "mllm_metrics.jsonl")
    interactions = interaction_summary(result.get("interaction_attempts") or [], result)
    trace_data = trace_summary(trace)
    timings = trace_data.pop("_timings")
    coverage = (topdown.get("coverage") or {}).get("exploration_coverage_ratio")
    spatial = {
        "coverage_ratio": numeric(coverage),
        "trajectory_samples": topdown.get("trajectory_samples"),
        "gt_oracle_path_complete": (topdown.get("gt_oracle_path") or {}).get("complete"),
        "gt_oracle_path_length_m": numeric((topdown.get("gt_oracle_path") or {}).get("length_m")),
        "gt_oracle_path_error": topdown.get("gt_oracle_path_error"),
        "navigation_path_length_m": numeric(result.get("navigation_path_length_m")),
        "reference_path_length_m": numeric(result.get("reference_path_length_m")),
        "gt_interaction_count": len(topdown.get("gt_interactions") or []),
        "actual_interaction_count": len(topdown.get("actual_interactions") or []),
        "nearest_actual_to_gt_interaction_m": nearest_interaction_error(topdown),
        "strict_target_xy": target_xy,
        "strict_target_object_name": target_object_name,
    }
    if spatial["navigation_path_length_m"] is not None and spatial["reference_path_length_m"] not in (None, 0):
        spatial["path_ratio"] = spatial["navigation_path_length_m"] / spatial["reference_path_length_m"]
    else:
        spatial["path_ratio"] = None
    diagnosis, evidence, recommendations = classify_episode(result, log_counts)
    add_contextual_findings(result, spatial, interactions, evidence, recommendations)
    relaxed_distance = manifest.get("relaxed_to_strict_target_distance_m")
    if result.get("goal_definition_relaxed_success") and relaxed_distance is not None:
        evidence.append(
            f"Relaxed instance {result.get('goal_definition_relaxed_instance_id')} is "
            f"{relaxed_distance:.2f} m from the frozen target coordinate."
        )
    stationary = manifest.get("longest_stationary_pose_run") or {}
    if int(stationary.get("frame_count") or 0) >= 100 and not result.get("nav_success"):
        evidence.append(
            f"Public frame poses contain a {stationary['frame_count']}-frame stationary run "
            f"({stationary.get('start_step')}–{stationary.get('end_step')})."
        )
        recommendations.append(
            "Trigger bounded recovery before a stationary pose run consumes 100 observation turns."
        )
    if interactions.get("state_contract_warning_count"):
        evidence.append(
            f"{interactions['state_contract_warning_count']} successful backend interaction(s) ended with a non-open/unknown state; inspect scan-vs-state semantics."
        )
        recommendations.append(
            "Separate backend macro completion from verified physical state transition in interaction metrics."
        )
    if mllm.get("error_count"):
        evidence.append(f"Structured MLLM metrics contain {mllm['error_count']} error(s).")
    recommendations[:] = list(dict.fromkeys(recommendations))
    frames = frame_samples(attempt_dir, offline)
    event_frames = event_frame_samples(attempt_dir, interactions, manifest)
    missing_raw = offline.get("missing_raw_step_indexes") or []
    missing_sim = offline.get("missing_sim_step_indexes") or []
    output_frames = int(offline.get("output_frame_count") or 0)
    exact_frames = int(offline.get("exact_step_match_count") or 0)
    sim_frames = int(offline.get("sim_frame_count") or 0)
    input_frames = int(offline.get("input_step_count") or 0)
    video_path = Path(str(row.get("six_panel_video") or episode_dir / "overview_6panel.mp4"))
    policy_termination = result.get("policy_termination") or {}
    terminal_detail = policy_termination.get("terminal_detail") or {}
    explicit_observation_turns = terminal_detail.get("observation_turn_count")
    observation_turn_count = int(
        explicit_observation_turns
        if explicit_observation_turns is not None
        else (input_frames or sim_frames or result.get("step_count") or 0)
    )
    observation_turn_limit = terminal_detail.get("observation_turn_limit")
    encoding_complete = bool(
        offline_path.is_file()
        and output_frames > 0
        and sim_frames == input_frames == output_frames == exact_frames
        and not missing_raw
        and not missing_sim
    )
    capture_complete = bool(sim_frames and sim_frames == observation_turn_count)
    recording_complete = bool(
        capture_complete
        and encoding_complete
        and offline_path.is_file()
        and video_path.is_file()
    )
    execution_state = debug_summary.get("semantic_decision_execution_state") or {}
    return {
        "episode_index": index,
        "group": group_for_index(index),
        "case_id": result.get("case_id") or row.get("case_id"),
        "house_index": result.get("house_index") or row.get("house_index"),
        "terminal_reason": result.get("terminal_reason") or row.get("terminal_reason"),
        "diagnosis": diagnosis,
        "diagnosis_evidence": evidence,
        "recommendations": recommendations,
        "success": bool(result.get("success")),
        "legacy_interaction_conditioned_success": bool(result.get("success")),
        "paper_sr_success": bool(result.get("nav_success")),
        "task_success": bool(result.get("task_success")),
        "nav_success": bool(result.get("nav_success")),
        "interaction_conditioned_success": bool(result.get("interaction_conditioned_success")),
        "required_interaction_success": bool(result.get("required_interaction_success")),
        "sequence_success": bool(result.get("sequence_success")),
        "interaction_requirement": result.get("interaction_requirement"),
        "goal_definition_relaxed_success": bool(result.get("goal_definition_relaxed_success")),
        "goal_definition_relaxed_instance_id": result.get("goal_definition_relaxed_instance_id"),
        "goal_definition_relaxed_reason": result.get("goal_definition_relaxed_reason"),
        "scoring_eligible": bool(result.get("scoring_eligible")),
        "step_count": int(result.get("step_count") or 0),
        "decision_trace_count": len(trace),
        "applied_action_step_count": int(result.get("applied_action_step_count") or 0),
        "no_fresh_action_count": int(result.get("no_fresh_action_count") or 0),
        "observation_turn_count": observation_turn_count,
        "observation_turn_limit": int(observation_turn_limit) if observation_turn_limit is not None else None,
        "observation_turn_count_source": (
            "policy_terminal_detail" if explicit_observation_turns is not None else "offline_recording"
        ),
        "episode_step_budget": int(result.get("episode_step_budget") or 0),
        "target_distance_m": numeric(result.get("target_distance_m")),
        "target_visibility_fraction": numeric(result.get("target_visibility_fraction")),
        "navigation_path_length_m": numeric(result.get("navigation_path_length_m")),
        "reference_path_length_m": numeric(result.get("reference_path_length_m")),
        "spl": numeric(result.get("spl")),
        "interaction_precision": numeric(result.get("interaction_precision_episode")),
        "episode_total_cost": numeric(result.get("episode_total_cost")),
        "elapsed_seconds": numeric(result.get("elapsed_seconds")),
        "runner_elapsed_seconds": numeric(row.get("elapsed_sec")),
        "post_evaluator_tail_seconds": (
            float(row["elapsed_sec"]) - float(result["elapsed_seconds"])
            if row.get("elapsed_sec") is not None and result.get("elapsed_seconds") is not None
            else None
        ),
        "policy_termination": policy_termination,
        "early_stop": result.get("early_stop") or {},
        "interaction": interactions,
        "mllm": mllm,
        "debug": {
            "summary_path": str(debug_summary_path),
            "present": debug_summary_path.is_file(),
            "final_image_step": debug_summary.get("final_image_step"),
            "stall_snapshot_count": int(debug_summary.get("stall_snapshot_count") or 0),
            "stall_snapshots": debug_summary.get("stall_snapshots") or [],
            "final_behavior_feedback": debug_summary.get("semantic_decision_behavior_feedback"),
            "final_selection": debug_summary.get("semantic_decision_selection"),
            "execution_state": {
                key: execution_state.get(key)
                for key in (
                    "state", "behavior_type", "candidate_id", "decision_id",
                    "elapsed_task_steps", "error", "latest_task_step_index",
                    "last_explore_feedback",
                )
            },
        },
        "graph_revision": graph_revision,
        "trace": trace_data,
        "_loop_timings": timings,
        "logs": {
            "counts": log_counts,
            "samples": log_samples,
            "files": scanned_logs,
        },
        "spatial": spatial,
        "recording": {
            "summary_path": str(offline_path),
            "summary_present": offline_path.is_file(),
            "video_present": video_path.is_file(),
            "sim_frame_count": sim_frames,
            "input_step_count": input_frames,
            "output_frame_count": output_frames,
            "exact_step_match_count": exact_frames,
            "missing_raw_step_count": len(missing_raw),
            "missing_sim_step_count": len(missing_sim),
            "exact_alignment": bool(output_frames and exact_frames == output_frames and not missing_raw and not missing_sim),
            "capture_complete": capture_complete,
            "encoding_complete": encoding_complete,
            "complete": recording_complete,
        },
        "visual": {
            "topdown": str(row.get("topdown") or episode_dir / "episode_topdown.png"),
            "sample_frames": [str(path) for path in frames],
            "event_frames": event_frames,
            "metrics": image_metrics(frames),
            "manifest": manifest,
        },
        "paths": {
            "episode_dir": str(episode_dir),
            "attempt_dir": str(attempt_dir),
            "result": str(result_path),
            "topdown_json": str(topdown_path),
            "frame_manifest": str(manifest_path),
            "debug_summary": str(debug_summary_path),
            "mllm_metrics": str(attempt_dir / "mllm_metrics.jsonl"),
            "graph_revision_events": str(graph_revision_path),
            "runner_log": str(attempt_dir / "runner.log"),
            "eval_log": str(attempt_dir / "eval.log"),
            "video": str(video_path),
        },
    }


def aggregate(
    episodes: list[dict[str, Any]],
    summary: dict[str, Any],
    round_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loop = [value for episode in episodes for value in episode.pop("_loop_timings")]
    mllm_latencies = [
        value for episode in episodes for value in episode["mllm"].pop("_latencies", [])
    ]
    mllm_queue_lags = [
        value for episode in episodes for value in episode["mllm"].pop("_queue_lags", [])
    ]
    groups: dict[str, Any] = {}
    paper_groups = ((round_contract or {}).get("paper_metrics") or {}).get("groups") or {}
    for name in ("all", "channel", "container", "mixed"):
        rows = episodes if name == "all" else [row for row in episodes if row["group"] == name]
        paper_key = "overall" if name == "all" else f"domain/{name}"
        paper = paper_groups.get(paper_key) or {}
        groups[name] = {
            "episode_count": len(rows),
            "paper_sr": numeric(paper.get("sr")) if paper else rate(rows, "paper_sr_success"),
            "legacy_ics": rate(rows, "legacy_interaction_conditioned_success"),
            # Compatibility aliases are explicit about their old meaning.
            "formal_success_rate": rate(rows, "legacy_interaction_conditioned_success"),
            "task_success_rate": rate(rows, "task_success"),
            "nav_success_rate": rate(rows, "nav_success"),
            "interaction_conditioned_success_rate": rate(rows, "interaction_conditioned_success"),
            "required_interaction_success_rate": rate(rows, "required_interaction_success"),
            "relaxed_success_rate": rate(rows, "goal_definition_relaxed_success"),
            "mean_spl": numeric(paper.get("spl")) if paper else mean(row["spl"] for row in rows),
            "interaction_precision": numeric(paper.get("ip")) if paper else mean(row["interaction_precision"] for row in rows),
            "mean_total_cost": numeric(paper.get("total_cost")) if paper else mean(row["episode_total_cost"] for row in rows),
            "mean_steps": mean(float(row["step_count"]) for row in rows),
            "mean_observation_turns": mean(float(row["observation_turn_count"]) for row in rows),
            "mean_coverage": mean(row["spatial"]["coverage_ratio"] for row in rows),
            "mean_runner_seconds": mean(row["runner_elapsed_seconds"] for row in rows),
        }
    starts = []
    finishes = []
    summary_rows = {int(row["episode_index"]): row for row in summary.get("episodes", [])}
    for episode in episodes:
        source = summary_rows.get(episode["episode_index"], {})
        for value, target in ((source.get("started_at"), starts), (source.get("finished_at"), finishes)):
            if value:
                target.append(datetime.fromisoformat(value))
    span = (max(finishes) - min(starts)).total_seconds() if starts and finishes else None
    total_steps = sum(row["step_count"] for row in episodes)
    return {
        "episode_count": len(episodes),
        "groups": groups,
        "terminal_reason_counts": dict(Counter(row["terminal_reason"] for row in episodes)),
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in episodes)),
        "recommendation_counts": dict(Counter(item for row in episodes for item in row["recommendations"])),
        "total_steps": total_steps,
        "scene_span_seconds": span,
        "throughput_steps_per_second": total_steps / span if span else None,
        "loop_timing": {
            "count": len(loop),
            "mean_seconds": mean(loop),
            "p50_seconds": percentile(loop, 0.50),
            "p95_seconds": percentile(loop, 0.95),
            "p99_seconds": percentile(loop, 0.99),
            "max_seconds": max(loop) if loop else None,
        },
        "canonical_round_contract": {
            "available": round_contract is not None,
            "schema_version": (round_contract or {}).get("schema_version"),
            "warnings": (round_contract or {}).get("warnings") or [],
            "paper_metric_schema_version": ((round_contract or {}).get("paper_metrics") or {}).get("paper_metric_schema_version"),
        },
        "mllm": {
            "call_count": sum(row["mllm"]["call_count"] for row in episodes),
            "error_count": sum(row["mllm"]["error_count"] for row in episodes),
            "error_counts": dict(sum((Counter(row["mllm"]["error_counts"]) for row in episodes), Counter())),
            "role_counts": dict(sum((Counter(row["mllm"]["role_counts"]) for row in episodes), Counter())),
            "latency_seconds": {
                "mean": mean(mllm_latencies),
                "p50": percentile(mllm_latencies, 0.50),
                "p95": percentile(mllm_latencies, 0.95),
                "p99": percentile(mllm_latencies, 0.99),
                "max": max(mllm_latencies) if mllm_latencies else None,
            },
            "queue_lag_seconds": {
                "mean": mean(mllm_queue_lags),
                "p50": percentile(mllm_queue_lags, 0.50),
                "p95": percentile(mllm_queue_lags, 0.95),
                "p99": percentile(mllm_queue_lags, 0.99),
                "max": max(mllm_queue_lags) if mllm_queue_lags else None,
            },
        },
        "log_signal_totals": dict(
            sum((Counter(row["logs"]["counts"]) for row in episodes), Counter())
        ),
        "graph_revision": {
            "file_count": sum(bool((row.get("graph_revision") or {}).get("present")) for row in episodes),
            "event_count": sum(int((row.get("graph_revision") or {}).get("event_count") or 0) for row in episodes),
            "event_counts": dict(sum(
                (Counter((row.get("graph_revision") or {}).get("event_counts") or {}) for row in episodes),
                Counter(),
            )),
            "state_transition_counts": dict(sum(
                (Counter((row.get("graph_revision") or {}).get("state_transition_counts") or {}) for row in episodes),
                Counter(),
            )),
            "invalid_row_count": sum(
                int((row.get("graph_revision") or {}).get("invalid_row_count") or 0) for row in episodes
            ),
        },
        "recording": {
            "video_count": sum(Path(row["paths"]["video"]).is_file() for row in episodes),
            "output_frame_count": sum(row["recording"]["output_frame_count"] for row in episodes),
            "exact_alignment_episode_count": sum(row["recording"]["exact_alignment"] for row in episodes),
            "capture_complete_episode_count": sum(
                bool(row["recording"].get("capture_complete")) for row in episodes
            ),
            "encoding_complete_episode_count": sum(
                bool(row["recording"].get("encoding_complete")) for row in episodes
            ),
            "complete_episode_count": sum(row["recording"]["complete"] for row in episodes),
            "missing_raw_step_count": sum(row["recording"]["missing_raw_step_count"] for row in episodes),
            "missing_sim_step_count": sum(row["recording"]["missing_sim_step_count"] for row in episodes),
        },
    }


def resource_telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    cpu_values: list[float] = []
    active_cpu_values: list[float] = []
    mem_used: list[float] = []
    mem_available: list[float] = []
    gpu: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"memory": [], "utilization": []})
    row_count = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.DictReader(stream):
            row_count += 1
            cpu = numeric(row.get("host_cpu_percent"))
            if cpu is not None:
                cpu_values.append(cpu)
                if int(float(row.get("active_episode_count") or 0)) > 0:
                    active_cpu_values.append(cpu)
            used = numeric(row.get("host_mem_used_mb"))
            available = numeric(row.get("host_mem_available_mb"))
            if used is not None:
                mem_used.append(used)
            if available is not None:
                mem_available.append(available)
            try:
                gpu_rows = json.loads(row.get("gpu_metrics_json") or "[]")
            except json.JSONDecodeError:
                gpu_rows = []
            for item in gpu_rows:
                index = int(item.get("index") or 0)
                memory = numeric(item.get("memory_used_mb"))
                utilization = numeric(item.get("utilization_gpu_percent"))
                if memory is not None:
                    gpu[index]["memory"].append(memory)
                if utilization is not None:
                    gpu[index]["utilization"].append(utilization)
    memory_trustworthy = bool(
        mem_used
        and mem_available
        and max(mem_used) >= 128.0
        and max(mem_used) <= max(mem_available) * 10.0
    )
    warnings = []
    if mem_used and not memory_trustworthy:
        warnings.append(
            "host_mem_used_mb is implausible relative to host_mem_available_mb; host memory statistics are reported as N/A"
        )
    return {
        "present": True,
        "path": str(path),
        "row_count": row_count,
        "host_cpu_percent": {
            "mean": mean(cpu_values),
            "active_mean": mean(active_cpu_values),
            "peak": max(cpu_values) if cpu_values else None,
        },
        "host_memory": {
            "trustworthy": memory_trustworthy,
            "mean_used_mb": mean(mem_used) if memory_trustworthy else None,
            "peak_used_mb": max(mem_used) if memory_trustworthy and mem_used else None,
            "mean_available_mb": mean(mem_available) if memory_trustworthy else None,
        },
        "gpus": {
            str(index): {
                "mean_memory_used_mb": mean(values["memory"]),
                "peak_memory_used_mb": max(values["memory"]) if values["memory"] else None,
                "mean_utilization_percent": mean(values["utilization"]),
                "peak_utilization_percent": max(values["utilization"]) if values["utilization"] else None,
            }
            for index, values in sorted(gpu.items())
        },
        "warnings": warnings,
    }


def build_reproduction_checks(episodes: list[dict[str, Any]], aggregate_data: dict[str, Any]) -> list[dict[str, Any]]:
    by_index = {row["episode_index"]: row for row in episodes}
    claim_rows = [row for row in episodes if row["terminal_reason"] == "target_claim_unverified"]
    relaxed_claims = [row["episode_index"] for row in claim_rows if row["goal_definition_relaxed_success"]]
    unsupported_claims = [row["episode_index"] for row in claim_rows if not row["goal_definition_relaxed_success"]]
    fresh_timer = [
        row["episode_index"] for row in episodes
        if row["diagnosis"] == "global_mission_timer_triggered_on_new_subgoal"
    ]
    no_candidate = [
        row["episode_index"] for row in episodes
        if row["diagnosis"] == "candidate_generation_exhausted_despite_raw_frontiers"
    ]
    local_stall = [
        row["episode_index"] for row in episodes
        if row["diagnosis"] == "portal_approach_exhausted_after_navigation_stagnation"
    ]
    tf_stall = [row["episode_index"] for row in episodes if row["diagnosis"] == "tf_action_lifecycle_stall"]
    state_warnings = [
        row["episode_index"] for row in episodes
        if row["interaction"]["state_contract_warning_count"]
    ]
    benchmark_required = [
        row["episode_index"] for row in episodes
        if row["terminal_reason"] == "target_found"
        and row["task_success"]
        and not row["required_interaction_success"]
        and row["interaction"]["attempt_count"] == 0
    ]
    route_missing = [
        row["episode_index"] for row in episodes
        if row["spatial"]["gt_oracle_path_complete"] is False
    ]
    runtime_tracebacks = sum(
        row["logs"]["counts"].get("runtime_traceback", 0) for row in episodes
    )
    fatal_process_signals = sum(
        row["logs"]["counts"].get(key, 0)
        for row in episodes for key in ("oom", "segfault")
    )
    return [
        {
            "id": "claim_instance_split",
            "status": "reproduced"
            if claim_rows and len(relaxed_claims) + len(unsupported_claims) == len(claim_rows)
            else "not_observed",
            "summary": "Target-claim failures split into relaxed same-category instance hits and unsupported claims.",
            "evidence": {"total": len(claim_rows), "relaxed_instance": relaxed_claims, "unsupported": unsupported_claims},
        },
        {
            "id": "raw_frontier_candidate_pipeline_gap",
            "status": "reproduced" if no_candidate else "not_observed",
            "summary": "Raw physical frontiers survived, but the executable proposal pipeline returned empty after bounded recovery.",
            "evidence": {"episodes": no_candidate},
        },
        {
            "id": "global_mission_timer_on_new_subgoal",
            "status": "reproduced" if fresh_timer else "not_observed",
            "summary": "Mission-age timeout fired while the selected subgoal elapsed counter was zero.",
            "evidence": {"episodes": fresh_timer},
        },
        {
            "id": "portal_approach_local_stagnation",
            "status": "reproduced" if local_stall else "not_observed",
            "summary": "Portal approach options exhausted after local navigation stagnation.",
            "evidence": {"episodes": local_stall},
        },
        {
            "id": "tf_action_lifecycle_stall",
            "status": "reproduced" if tf_stall else "not_observed",
            "summary": "Low-displacement cross-subgoal stall coincided with runtime TF and missing-goal/rear-goal lifecycle errors.",
            "evidence": {"episodes": tf_stall},
        },
        {
            "id": "interaction_backend_vs_state_contract",
            "status": "reproduced" if state_warnings else "not_observed",
            "summary": "Successful backend macros sometimes ended with closed/unknown/unavailable state; this is a contract ambiguity, not automatically a failed interaction.",
            "evidence": {"episodes": state_warnings},
        },
        {
            "id": "required_interaction_benchmark_anomaly",
            "status": "reproduced" if benchmark_required else "not_observed",
            "summary": "A required-interaction episode reached the target without any interaction.",
            "evidence": {"episodes": benchmark_required},
        },
        {
            "id": "oracle_route_availability",
            "status": "reproduced" if route_missing else "not_observed",
            "summary": "Recomputed top-down oracle path is incomplete for a subset; frozen reference metrics remain separate.",
            "evidence": {"episodes": route_missing},
        },
        {
            "id": "recording_two_layer_contract",
            "status": "reproduced"
            if aggregate_data["recording"]["encoding_complete_episode_count"] == len(episodes)
            else "unexpected",
            "summary": "All acquired frames encode exactly, while raw acquisition completeness is reported separately.",
            "evidence": aggregate_data["recording"],
        },
        {
            "id": "runtime_failure_signal_audit",
            "status": "reproduced" if fatal_process_signals == 0 else "unexpected",
            "summary": "No fatal OOM/segfault was found. Recoverable runtime tracebacks and shutdown tracebacks are counted separately rather than hidden.",
            "evidence": {
                "fatal_process_signals": fatal_process_signals,
                "runtime_tracebacks": runtime_tracebacks,
                "shutdown_tracebacks": aggregate_data["log_signal_totals"].get("shutdown_traceback", 0),
            },
        },
        {
            "id": "paper_metric_semantics",
            "status": "reproduced",
            "summary": "Canonical paper SR uses nav_success; legacy ICS uses persisted result.success.",
            "evidence": {
                "paper_sr": aggregate_data["groups"]["all"]["paper_sr"],
                "legacy_ics": aggregate_data["groups"]["all"]["legacy_ics"],
            },
        },
    ]


def apply_audit_notes(
    episodes: list[dict[str, Any]], evaluation_dir: Path, notes_path: Path | None
) -> dict[str, Any] | None:
    if notes_path is None:
        return None
    notes = read_json(notes_path, None)
    if not isinstance(notes, dict):
        raise ValueError(f"Invalid audit notes: {notes_path}")
    expected = notes.get("source_evaluation_dir")
    if expected and Path(expected).expanduser().resolve() != evaluation_dir:
        raise ValueError(
            f"Audit notes were created for {expected}, not {evaluation_dir}"
        )
    by_index = notes.get("episodes") or {}
    for episode in episodes:
        note = by_index.get(str(episode["episode_index"])) or by_index.get(episode["episode_index"])
        if isinstance(note, dict):
            episode["visual_audit"] = note
    return notes


def evidence_index(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "interactive_nav_offline_evidence_index_v1",
        "episodes": {
            f"{row['episode_index']:04d}": {
                "diagnosis": row["diagnosis"],
                "result": row["paths"]["result"],
                "result_json_pointers": [
                    "/result/terminal_reason",
                    "/result/policy_termination",
                    "/result/interaction_attempts",
                    "/result/early_stop",
                ],
                "topdown_json": row["paths"]["topdown_json"],
                "topdown_image": row["visual"]["topdown"],
                "frame_manifest": row["paths"]["frame_manifest"],
                "sample_frames": row["visual"]["sample_frames"],
                "event_frames": row["visual"].get("event_frames", []),
                "video": row["paths"]["video"],
                "debug_summary": row["paths"]["debug_summary"],
                "graph_revision_events": row["paths"]["graph_revision_events"],
                "mllm_metrics": row["paths"]["mllm_metrics"],
                "logs": row["logs"]["files"],
                "interaction_steps": [
                    item.get("decision_step") for item in row["interaction"]["attempts"]
                    if item.get("decision_step") is not None
                ],
                "relaxed_instance_frame": (
                    (row["visual"]["manifest"].get("relaxed_target_instance") or {}).get("max_mask_step")
                ),
            }
            for row in episodes
        },
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def relative_delta(value: float | None, baseline: float | None) -> str:
    if value is None or baseline in (None, 0):
        return "N/A"
    return f"{(value / baseline - 1) * 100:+.1f}%"


def render_markdown(analysis: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    aggregate_data = analysis["aggregate"]
    episodes = analysis["episodes"]
    lines = [
        "# InteractiveNav offline benchmark analysis",
        "",
        f"- Schema: `{SCHEMA_VERSION}`",
        f"- Evaluation directory: `{analysis['evaluation_dir']}`",
        f"- Episodes analyzed: {len(episodes)}",
        "- Method: deterministic offline parsing of committed results, traces, logs, maps, and recording manifests; no simulator was run.",
        "",
        "## Success metrics",
        "",
        "| Group | n | Paper SR (Nav) | Legacy ICS | TaskSR | ISR | Same-category diagnostic | SPL | IP | Total cost | Mean trace steps | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "channel", "container", "mixed"):
        group = aggregate_data["groups"][name]
        lines.append(
            f"| {name} | {group['episode_count']} | {percent(group['paper_sr'])} | "
            f"{percent(group['legacy_ics'])} | {percent(group['task_success_rate'])} | "
            f"{percent(group['required_interaction_success_rate'])} | {percent(group['relaxed_success_rate'])} | "
            f"{fmt(group['mean_spl'])} | {fmt(group['interaction_precision'])} | "
            f"{fmt(group['mean_total_cost'])} | {fmt(group['mean_steps'], 1)} | {percent(group['mean_coverage'])} |"
        )
    timing = aggregate_data["loop_timing"]
    lines += [
        "",
        "## Performance and artifact integrity",
        "",
        f"- Scene span: {fmt(aggregate_data['scene_span_seconds'], 1)} s",
        f"- Total steps / throughput: {aggregate_data['total_steps']} / {fmt(aggregate_data['throughput_steps_per_second'])} step/s",
        f"- Loop mean / P50 / P95 / P99: {fmt(timing['mean_seconds'])} / {fmt(timing['p50_seconds'])} / {fmt(timing['p95_seconds'])} / {fmt(timing['p99_seconds'])} s",
        f"- Raw capture complete: {aggregate_data['recording'].get('raw_capture_complete_episode_count', 'N/A')}/{len(episodes)}; shortfalls: {aggregate_data['recording'].get('raw_capture_shortfalls', [])}",
        f"- Final videos: {aggregate_data['recording']['video_count']}/{len(episodes)}; offline exact-alignment episodes: {aggregate_data['recording']['exact_alignment_episode_count']}/{len(episodes)}",
        f"- Recorded frames: {aggregate_data['recording']['output_frame_count']}; missing raw/sim steps: {aggregate_data['recording']['missing_raw_step_count']}/{aggregate_data['recording']['missing_sim_step_count']}",
        f"- MLLM calls/errors: {aggregate_data['mllm']['call_count']}/{aggregate_data['mllm']['error_count']}; latency P50/P95/P99: {fmt(aggregate_data['mllm']['latency_seconds']['p50'])}/{fmt(aggregate_data['mllm']['latency_seconds']['p95'])}/{fmt(aggregate_data['mllm']['latency_seconds']['p99'])} s",
        "",
        "## Terminal reasons and diagnoses",
        "",
        "### Terminal reason counts",
        "",
    ]
    for key, value in sorted(aggregate_data["terminal_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "### Offline diagnosis counts", ""]
    for key, value in sorted(aggregate_data["diagnosis_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {value}")

    lines += ["", "### Prioritized modification opinions (not applied)", ""]
    for key, value in sorted(aggregate_data["recommendation_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {value} episode(s): {key}")

    lines += [
        "",
        "## All episodes",
        "",
        "| Episode | Group | Terminal | Diagnosis | Paper/Legacy/ISR | Obs/Applied/Trace/Budget | No-fresh | Distance | Visibility | Interactions | Interaction failures | Coverage | Bridge timeout→noop |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in episodes:
        interaction = row["interaction"]
        lines.append(
            f"| {row['episode_index']:04d} | {row['group']} | `{row['terminal_reason']}` | `{row['diagnosis']}` | "
            f"{int(row['paper_sr_success'])}/{int(row['legacy_interaction_conditioned_success'])}/{int(row['required_interaction_success'])} | "
            f"{row['observation_turn_count']}/{row['applied_action_step_count']}/{row['decision_trace_count']}/{row['episode_step_budget']} | {row['no_fresh_action_count']} | "
            f"{fmt(row['target_distance_m'], 2)} | {fmt(row['target_visibility_fraction'], 5)} | "
            f"{interaction['success_count']}/{interaction['attempt_count']} | "
            f"{', '.join(f'{key}:{value}' for key, value in sorted(interaction['failure_reasons'].items())) or '-'} | "
            f"{percent(row['spatial']['coverage_ratio'])} | "
            f"{row['logs']['counts'].get('action_timeout', 0)} |"
        )

    lines += ["", "## Evidence and recommendations by terminal reason", ""]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[row["terminal_reason"]].append(row)
    for reason, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        lines += [f"### `{reason}` ({len(rows)})", ""]
        for row in rows:
            lines.append(
                f"- **{row['episode_index']:04d}** `{row['diagnosis']}` — "
                + " ".join(row["diagnosis_evidence"])
            )
            lines.append(
                f"  Evidence: [result]({row['paths']['result']}), [topdown]({row['visual']['topdown']}), "
                f"[eval.log]({row['paths']['eval_log']}), [video]({row['paths']['video']})."
            )
        recommendations = list(dict.fromkeys(item for row in rows for item in row["recommendations"]))
        if recommendations:
            lines += ["", "Recommended changes (not applied):"]
            lines.extend(f"- {item}" for item in recommendations)
        lines.append("")

    lines += ["## Targeted offline reproductions", ""]
    lines += [
        "| Check | Status | Result |",
        "| --- | --- | --- |",
    ]
    for check in analysis.get("offline_reproduction_checks", []):
        lines.append(
            f"| `{check['id']}` | `{check['status']}` | {check['summary']} Evidence: `{json.dumps(check['evidence'], ensure_ascii=False)}` |"
        )

    lines += [
        "",
        "## Visual evidence",
        "",
        "Each generated episode panel contains the top-down map followed by first, middle, and final camera frames.",
        "The JSON also records brightness/dark-frame ratios and first-to-last frame change for reproducible triage.",
        "See `visual_contact_sheet.jpg` and `episode_visuals/` next to this report.",
        "",
        "## Offline reproduction commands",
        "",
        "```bash",
        f"python scripts/InteractiveNav/analyze_benchmark_run.py {analysis['evaluation_dir']} --output-dir {analysis['output_dir']}",
        "pytest -q scripts/InteractiveNav/test_analyze_benchmark_run.py",
        "```",
        "",
        "No algorithm source was changed and no simulation was launched by these commands.",
    ]

    if baseline:
        lines += ["", "## Supplied performance baseline comparison", ""]
        current = {
            "scene_span_seconds": aggregate_data["scene_span_seconds"],
            "weighted_loop_seconds": timing["mean_seconds"],
            "throughput_steps_per_second": aggregate_data["throughput_steps_per_second"],
            "p50_seconds": timing["p50_seconds"],
            "p95_seconds": timing["p95_seconds"],
            "p99_seconds": timing["p99_seconds"],
        }
        lines += [
            "The baselines use fixed 200-step budgets, while this run uses dynamic budgets; wall-time deltas are descriptive, not controlled worker-scaling estimates.",
            "",
            "| Baseline | Span delta | Loop delta | Throughput delta | P50 delta | P95 delta | P99 delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for item in baseline.get("runs", []):
            metrics = item.get("metrics", {})
            lines.append(
                f"| {item['name']} | {relative_delta(current['scene_span_seconds'], metrics.get('scene_span_seconds'))} | "
                f"{relative_delta(current['weighted_loop_seconds'], metrics.get('weighted_loop_seconds'))} | "
                f"{relative_delta(current['throughput_steps_per_second'], metrics.get('throughput_steps_per_second'))} | "
                f"{relative_delta(current['p50_seconds'], metrics.get('p50_seconds'))} | "
                f"{relative_delta(current['p95_seconds'], metrics.get('p95_seconds'))} | "
                f"{relative_delta(current['p99_seconds'], metrics.get('p99_seconds'))} |"
            )
    return "\n".join(lines) + "\n"


DIAGNOSIS_ZH = {
    "verified_success": "完成严格目标与交互条件",
    "formal_success_blocked_by_interaction_requirement": "到达目标但交互条件未满足，疑似任务交互必要性与实际路径不一致",
    "target_found_metric_inconsistency": "target_found 与评分字段不一致",
    "wrong_instance_same_category_target_claim": "找到同类别但非冻结目标实例",
    "near_target_without_public_visual_evidence": "接近目标但缺少可验证视觉证据",
    "ungrounded_or_premature_target_claim": "目标 claim 缺少位置/可见性支撑",
    "budget_exhaustion_after_wrong_interaction_binding": "错误或不可用交互绑定后耗尽观察预算",
    "observation_budget_exhausted_without_goal": "观察轮次预算耗尽且未验证目标",
    "candidate_generation_exhausted_despite_raw_frontiers": "仍有原始 frontier，但候选转换管线产出为空",
    "candidate_generation_exhausted_after_recovery": "有限恢复后无可执行候选",
    "global_mission_timer_triggered_on_new_subgoal": "全局 mission 无进展计时在新子目标刚开始时触发",
    "portal_approach_exhausted_after_navigation_stagnation": "portal 接近导航停滞后耗尽锚点",
    "semantic_mission_no_progress": "语义任务无进展终止",
    "tf_action_lifecycle_stall": "TF 时间戳与 action goal 生命周期故障导致跨子目标停滞",
    "repeated_navigation_failures_with_low_displacement": "多子目标导航失败且几乎无位移",
}


def _diagnosis_zh(row: dict[str, Any]) -> str:
    return DIAGNOSIS_ZH.get(row["diagnosis"], row["diagnosis"])


def _policy_detail(row: dict[str, Any]) -> dict[str, Any]:
    termination = row.get("policy_termination") or {}
    terminal = termination.get("terminal_detail") or {}
    latest = ((termination.get("latest_goal_status") or {}).get("detail") or {})
    keys = (
        "reason", "failure_reason", "mission_elapsed_task_steps",
        "mission_timeout_task_steps", "subgoal_elapsed_task_steps",
        "subgoal_timeout_task_steps", "interaction_approach_options_exhausted",
        "failure_stage", "candidate_id", "decision_id",
    )
    result = {"reported_reason": terminal.get("reported_reason")}
    result.update({key: latest.get(key) for key in keys if latest.get(key) is not None})
    context = latest.get("exploration_context") or {}
    if context:
        result["exploration_context"] = {
            key: context.get(key)
            for key in (
                "raw_frontier_cluster_count", "raw_frontier_cell_count",
                "raw_material_frontier_count", "proposal_count",
                "connected_unknown_area_present", "frontier_exhausted",
            )
            if context.get(key) is not None
        }
    return result


def render_markdown_zh(analysis: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    agg = analysis["aggregate"]
    episodes = analysis["episodes"]
    overall = agg["groups"]["all"]
    recording = agg["recording"]
    mllm = agg["mllm"]
    graph = agg["graph_revision"]
    resource = analysis.get("resource_telemetry") or {}
    provenance = analysis.get("provenance") or {}
    episode_count = len(episodes)
    paper_success_count = sum(bool(row["paper_sr_success"]) for row in episodes)
    legacy_success_count = sum(bool(row["legacy_interaction_conditioned_success"]) for row in episodes)
    scoring_count = sum(bool(row["scoring_eligible"]) for row in episodes)
    runtime_crash_count = sum(
        int(row["logs"]["counts"].get(key, 0))
        for row in episodes
        for key in ("runtime_traceback", "oom", "segfault")
    )
    target_claims = [row for row in episodes if row["terminal_reason"] == "target_claim_unverified"]
    relaxed_claims = [row for row in target_claims if row["goal_definition_relaxed_success"]]
    unsupported_claims = [row["episode_index"] for row in target_claims if not row["goal_definition_relaxed_success"]]
    policy_stalls = [row for row in episodes if row["terminal_reason"] == "policy_exploration_stalled"]
    policy_stall_breakdown = {
        diagnosis: [row["episode_index"] for row in policy_stalls if row["diagnosis"] == diagnosis]
        for diagnosis in sorted({row["diagnosis"] for row in policy_stalls})
    }
    tf_stalls = [
        row["episode_index"] for row in episodes if row["diagnosis"] == "tf_action_lifecycle_stall"
    ]
    mllm_error_episodes = [
        row["episode_index"] for row in episodes if row["mllm"]["error_count"]
    ]
    checks_by_id = {
        check["id"]: check for check in analysis.get("offline_reproduction_checks", [])
    }
    benchmark_flags = {
        key: (checks_by_id.get(key, {}).get("evidence") or {}).get("episodes", [])
        for key in ("required_interaction_benchmark_anomaly", "oracle_route_availability")
    }
    lines = [
        "# InteractiveNav Benchmark 完整离线分析记录",
        "",
        "## 1. 范围、约束与方法",
        "",
        f"- 输入：`{analysis['evaluation_dir']}`，共 {len(episodes)} 场。",
        "- 本报告由 `scripts/InteractiveNav/analyze_benchmark_run.py` 从既有 result、trace、ROS 节点日志、MLLM JSONL、top-down、逐帧 manifest 与录像摘要确定性生成。",
        "- 本次只做离线读取、图像取样与合成/保存数据回放测试；未启动 ROS、MuJoCo、Qwen 或实际仿真，未修改算法。",
        "- 既有 `evaluation/v3_round_summary.py` 负责 latest-attempt 与论文指标契约；本脚本只补充终止归因、日志、视觉和证据索引。",
        "",
        "## 2. 一页结论",
        "",
        f"- 论文口径 Paper SR（`nav_success`）：**{percent(overall['paper_sr'])}（{paper_success_count}/{episode_count}）**；历史 Legacy ICS（`result.success`）：**{percent(overall['legacy_ics'])}（{legacy_success_count}/{episode_count}）**。",
        f"- 终止原因：{json.dumps(agg['terminal_reason_counts'], ensure_ascii=False)}。可评分 {scoring_count}/{episode_count}；runtime traceback/OOM/segfault 信号共 {runtime_crash_count}。",
        f"- `target_claim_unverified` 共 {len(target_claims)} 场，其中 {len(relaxed_claims)} 场通过同类别 relaxed verifier；没有 relaxed 支持的 episode：`{unsupported_claims}`。",
        f"- `policy_exploration_stalled` 共 {len(policy_stalls)} 场，结构化根因拆分：`{json.dumps(policy_stall_breakdown, ensure_ascii=False)}`。",
        f"- TF/action lifecycle 低位移停滞 episode：`{tf_stalls}`；这类应先检查底层导航链路，再调整高层探索参数。",
        f"- benchmark/路线有效性离线疑点：`{json.dumps(benchmark_flags, ensure_ascii=False)}`；这些是审计线索，不直接断言 benchmark 数据错误。",
        "",
        "## 3. 指标口径",
        "",
        "| 分组 | n | Paper SR/Nav | Legacy ICS | ISR | SPL | IP | Total Cost | 同类对象诊断通过率（非评分） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("all", "channel", "container", "mixed"):
        group = agg["groups"][name]
        lines.append(
            f"| {name} | {group['episode_count']} | {percent(group['paper_sr'])} | "
            f"{percent(group['legacy_ics'])} | {percent(group['required_interaction_success_rate'])} | "
            f"{fmt(group['mean_spl'])} | {fmt(group['interaction_precision'])} | "
            f"{fmt(group['mean_total_cost'])} | {percent(group['relaxed_success_rate'])} |"
        )
    lines += [
        "",
        "说明：`target_visibility_fraction` 是终止时 evaluator-private GT 可见度，不等同 claim 当时的公开证据；coverage 是全局探索覆盖率；bridge timeout→noop 是等待指令超时日志事件，不等同动作执行失败。",
        "",
        "## 4. 数据完整性、性能和基础设施",
        "",
        f"- 场景总跨度 {fmt(agg['scene_span_seconds'], 1)} s；trace 共 {agg['total_steps']} 条，批次吞吐 {fmt(agg['throughput_steps_per_second'])} trace event/s。",
        f"- trace timing mean/P50/P95/P99/max：{fmt(agg['loop_timing']['mean_seconds'])}/{fmt(agg['loop_timing']['p50_seconds'])}/{fmt(agg['loop_timing']['p95_seconds'])}/{fmt(agg['loop_timing']['p99_seconds'])}/{fmt(agg['loop_timing']['max_seconds'])} s。",
        f"- 原始采集完整 {recording.get('raw_capture_complete_episode_count')}/{len(episodes)}；短缺：`{json.dumps(recording.get('raw_capture_shortfalls', []), ensure_ascii=False)}`。",
        f"- 对已经采集的帧，离线编码 exact 对齐 {recording['exact_alignment_episode_count']}/{len(episodes)}，最终视频 {recording['video_count']}/{len(episodes)}，共 {recording['output_frame_count']} 帧，missing raw/sim={recording['missing_raw_step_count']}/{recording['missing_sim_step_count']}。",
        f"- 日志信号汇总：`{json.dumps(agg['log_signal_totals'], ensure_ascii=False)}`。shutdown traceback 与运行期 traceback 分开统计。",
        f"- 语义图 revision：{graph['file_count']}/{episode_count} 场有历史，共 {graph['event_count']} 个事件；事件类型 `{json.dumps(graph['event_counts'], ensure_ascii=False)}`，状态转移 `{json.dumps(graph['state_transition_counts'], ensure_ascii=False)}`，坏行 {graph['invalid_row_count']}。",
        f"- CPU mean/active mean/peak：{fmt((resource.get('host_cpu_percent') or {}).get('mean'))}/{fmt((resource.get('host_cpu_percent') or {}).get('active_mean'))}/{fmt((resource.get('host_cpu_percent') or {}).get('peak'))}%。",
        f"- 主机内存遥测可信：`{(resource.get('host_memory') or {}).get('trustworthy')}`；警告：`{json.dumps(resource.get('warnings', []), ensure_ascii=False)}`。",
        "",
        "## 5. Qwen/MLLM 调用健康度",
        "",
        f"- 总调用 {mllm['call_count']}，错误 {mllm['error_count']}；角色分布：`{json.dumps(mllm['role_counts'], ensure_ascii=False)}`。",
        f"- latency mean/P50/P95/P99/max：{fmt(mllm['latency_seconds']['mean'])}/{fmt(mllm['latency_seconds']['p50'])}/{fmt(mllm['latency_seconds']['p95'])}/{fmt(mllm['latency_seconds']['p99'])}/{fmt(mllm['latency_seconds']['max'])} s。",
        f"- queue lag P95/P99/max：{fmt(mllm['queue_lag_seconds']['p95'])}/{fmt(mllm['queue_lag_seconds']['p99'])}/{fmt(mllm['queue_lag_seconds']['max'])} s；出现错误的 episode：`{mllm_error_episodes}`。是否构成主因需结合各场终止链判断。",
        "",
        "## 6. 终止根因分类",
        "",
        "| 根因诊断 | 场数 | Episode |",
        "| --- | ---: | --- |",
    ]
    for diagnosis, count in sorted(agg["diagnosis_counts"].items(), key=lambda item: (-item[1], item[0])):
        ids = [f"{row['episode_index']:04d}" for row in episodes if row["diagnosis"] == diagnosis]
        lines.append(f"| {_diagnosis_zh(next(row for row in episodes if row['diagnosis'] == diagnosis))} (`{diagnosis}`) | {count} | {', '.join(ids)} |")

    lines += [
        "",
        f"## 7. {episode_count} 场逐场审计表",
        "",
        "| EP | 类别/目标 | 终止与根因 | Paper/Legacy/ISR | Obs/Applied/Trace/Budget | 路径 actual/ref | 覆盖 | 交互 correct/attempt | 视觉/场内结论 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in episodes:
        target = row["spatial"].get("strict_target_object_name") or "?"
        note = row.get("visual_audit") or {}
        conclusion = note.get("conclusion_zh") or _diagnosis_zh(row)
        lines.append(
            f"| {row['episode_index']:04d} | {row['group']} / `{target}` | `{row['terminal_reason']}`；{_diagnosis_zh(row)} | "
            f"{int(row['paper_sr_success'])}/{int(row['legacy_interaction_conditioned_success'])}/{int(row['required_interaction_success'])} | "
            f"{row['observation_turn_count']}/{row['applied_action_step_count']}/{row['decision_trace_count']}/{row['episode_step_budget']} | "
            f"{fmt(row['spatial']['navigation_path_length_m'], 2)}/{fmt(row['spatial']['reference_path_length_m'], 2)} | "
            f"{percent(row['spatial']['coverage_ratio'])} | {row['interaction']['correct_count']}/{row['interaction']['attempt_count']} | {conclusion} |"
        )

    lines += ["", "## 8. 定向离线复现", "", "| 检查 | 结果 | 断言与证据 |", "| --- | --- | --- |"]
    for check in analysis.get("offline_reproduction_checks", []):
        lines.append(
            f"| `{check['id']}` | `{check['status']}` | {check['summary']} `{json.dumps(check['evidence'], ensure_ascii=False)}` |"
        )

    lines += [
        "",
        "## 9. 修改建议（本次均未实施）",
        "",
        "| 优先级 | 范围 | 影响场景 | 建议 | 根因置信度 | 最小离线回归 |",
        "| --- | --- | --- | --- | --- | --- |",
        "| P0 | evaluator/任务定义 | 0007,1002,1003,1007,2002,2006,2008,2009 | 明确类别指令究竟按冻结实例还是任意同类实例计分；同时双报 exact 与 category diagnostic，不能混为 SR。 | 高 | 保存的 strict/relaxed ID 与坐标距离断言 |",
        "| P0 | 目标匹配 | 0008,1007 | 使用 ontology/token boundary，禁止 atomizer↔soap bottle、potato↔pot 的宽松子串匹配。 | 高/中 | 目标 matcher 纯函数 fixture |",
        "| P0 | mission progress | 1004,2003,2005 | 审计有效进展事件是否重置全局 timer；新可执行子目标 elapsed=0 时不得仅因旧 mission age 终止。 | 中高 | 回放 latest_goal_status.detail |",
        "| P0 | 导航基础设施 | 1008 | 修复 TF stamp、missing action goal 与 rear-goal heading 链路；底层健康后再调探索。 | 高 | 回放已保存 TF/action 状态序列 |",
        "| P0 | benchmark 有效性 | 0006,1002,1006,1007 | 验证 required edge 是否真正阻断所有合法路径，并审计 oracle/reference path。 | 中高 | 轨迹与 GT edge/top-down 几何契约 |",
        "| P1 | 交互 anchor/face | 1005,1009,2000,2002 | wrong-face 后使用几何不同的备选 anchor，并保存排除原因；区分 macro 完成与物理状态转换。 | 高 | 1009 正例与 1005 反例对照 |",
        "| P1 | mixed 阶段规划 | 2001,2003–2009 | channel 完成后显式推进到正确 container，过滤无关 dresser/container。 | 中高 | 保存候选序列离线评分 |",
        "| P2 | 可观测性 | 全部 | 将 frontier→proposal 过滤、timer reset、claim 帧、interaction 前后帧做结构化持久化。 | 高 | 分析器 artifact contract 测试 |",
        "",
        "## 10. 每场详细证据",
        "",
    ]
    for row in episodes:
        note = row.get("visual_audit") or {}
        manifest = row["visual"]["manifest"]
        lines += [
            f"### Episode {row['episode_index']:04d}",
            "",
            f"- case/house/domain：`{row['case_id']}` / `{row['house_index']}` / `{row['group']}`。",
            f"- 结论：**{note.get('conclusion_zh') or _diagnosis_zh(row)}**。置信度：`{note.get('confidence', '由结构化规则判定')}`。",
            f"- 计数：observation={row['observation_turn_count']}，applied={row['applied_action_step_count']}，trace={row['decision_trace_count']}，budget={row['episode_step_budget']}，no-fresh={row['no_fresh_action_count']}。",
            f"- target：distance={fmt(row['target_distance_m'], 3)} m，terminal visibility={fmt(row['target_visibility_fraction'], 6)}；relaxed={row['goal_definition_relaxed_success']} / `{row['goal_definition_relaxed_instance_id']}`，relaxed-to-strict={fmt(manifest.get('relaxed_to_strict_target_distance_m'), 3)} m。",
            f"- 路径/覆盖：actual={fmt(row['spatial']['navigation_path_length_m'], 3)} m，reference={fmt(row['spatial']['reference_path_length_m'], 3)} m，topdown oracle complete={row['spatial']['gt_oracle_path_complete']}，coverage={percent(row['spatial']['coverage_ratio'])}。",
            f"- policy detail：`{json.dumps(_policy_detail(row), ensure_ascii=False)}`。",
            f"- graph revision：events={row['graph_revision'].get('event_count')}，last_revision={row['graph_revision'].get('last_graph_revision')}，target-category nodes=`{row['graph_revision'].get('target_category_node_ids', [])}`，state transitions=`{json.dumps(row['graph_revision'].get('state_transition_counts', {}), ensure_ascii=False)}`。",
            f"- interactions：`{json.dumps(row['interaction']['attempts'], ensure_ascii=False)}`。",
            f"- 运行日志信号：`{json.dumps(row['logs']['counts'], ensure_ascii=False)}`；MLLM calls/errors={row['mllm']['call_count']}/{row['mllm']['error_count']}。",
            f"- 结构化证据：{'; '.join(row['diagnosis_evidence'])}",
            f"- 文件：[result]({row['paths']['result']}) · [topdown]({row['visual']['topdown']}) · [frame manifest]({row['paths']['frame_manifest']}) · [debug summary]({row['paths']['debug_summary']}) · [video]({row['paths']['video']})。",
            "",
        ]
    lines += [
        "## 11. 可复现命令与限制",
        "",
        "```bash",
        f"python scripts/InteractiveNav/analyze_benchmark_run.py {analysis['evaluation_dir']} --output-dir {analysis['output_dir']}" + (f" --audit-notes {analysis['audit_notes']['path']}" if analysis['audit_notes']['path'] else ""),
        "PYTHONDONTWRITEBYTECODE=1 /home/ldl/conda_envs/mlspaces/bin/python -m pytest -q scripts/InteractiveNav/test_analyze_benchmark_run.py",
        "```",
        "",
        f"- 分析器 SHA-256：`{(provenance.get('analyzer') or {}).get('sha256', 'N/A')}`。",
        f"- 输入文件 SHA-256：`{json.dumps({name: item.get('sha256') for name, item in (provenance.get('inputs') or {}).items() if item.get('present')}, ensure_ascii=False)}`。",
        "",
        "限制：视觉语义结论来自结构化 GT/manifest 与人工画面复核注释；brightness/MAD 或终止时 GT 可见度不能替代 claim 当时的公开视觉证据。因本次禁止运行仿真，修改建议只给出离线复现与未来回归断言，没有实施算法修复。",
    ]
    return "\n".join(lines) + "\n"


def build_visuals(episodes: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required unless --no-images is used") from exc
    visual_dir = output_dir / "episode_visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for row in episodes:
        canvas = Image.new("RGB", (1580, 620), "white")
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{row['episode_index']:04d} {row['group']} | {row['terminal_reason']} | {row['diagnosis']} | "
            f"obs/applied/trace {row['observation_turn_count']}/{row['applied_action_step_count']}/{row['decision_trace_count']} "
            f"| cov {percent(row['spatial']['coverage_ratio'])}"
        )
        draw.text((8, 8), title, fill="black")
        samples = row["visual"]["sample_frames"]
        candidates: list[tuple[str, str]] = []
        if samples:
            candidates.append(("initial", samples[0]))
            candidates.append(("terminal", samples[-1]))
        for item in row["visual"].get("event_frames", []):
            if item["kind"].startswith("interaction_at:") or item["kind"].startswith("relaxed_claim"):
                candidates.append((f"{item['kind']}@{item['step']}", item["path"]))
        if len(candidates) < 4 and len(samples) >= 2:
            candidates.append(("middle", samples[len(samples) // 2]))
        selected = []
        seen_paths = set()
        for label, source in candidates:
            if source not in seen_paths:
                selected.append((label, Path(source)))
                seen_paths.add(source)
            if len(selected) == 4:
                break
        sources = [("topdown", Path(row["visual"]["topdown"]))] + selected
        boxes = [
            (0, 35, 500, 540),
            (520, 55, 250, 188), (780, 55, 250, 188),
            (1040, 55, 250, 188), (1300, 55, 250, 188),
        ]
        for (label, source), (x, y, width, height) in zip(sources, boxes):
            try:
                image = Image.open(source).convert("RGB")
            except OSError:
                continue
            fitted = ImageOps.contain(image, (width, height))
            canvas.paste(fitted, (x + (width - fitted.width) // 2, y + (height - fitted.height) // 2))
            draw.text((x + 4, y + height + 3), label, fill="black")
        evidence = (
            f"dist={fmt(row['target_distance_m'], 2)}m vis={fmt(row['target_visibility_fraction'], 5)} "
            f"interactions={row['interaction']['success_count']}/{row['interaction']['attempt_count']} "
            f"no_fresh={row['no_fresh_action_count']} action_timeout={row['logs']['counts'].get('action_timeout', 0)}"
        )
        draw.text((520, 285), evidence, fill="black")
        for line_index, line in enumerate(row["diagnosis_evidence"][:3]):
            draw.text((520, 320 + line_index * 35), line[:145], fill="black")
        path = visual_dir / f"episode_{row['episode_index']:04d}.jpg"
        canvas.save(path, quality=88)
        panels.append(canvas.resize((790, 310)))
        row["visual"]["analysis_panel"] = str(path)
    columns = 2
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new("RGB", (columns * 790, rows * 310), (235, 235, 235))
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * 790, (index // columns) * 310))
    sheet.save(output_dir / "visual_contact_sheet.jpg", quality=90)


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fields = [
        "episode_index", "group", "case_id", "terminal_reason", "diagnosis",
        "paper_sr_success", "legacy_interaction_conditioned_success", "task_success",
        "nav_success", "required_interaction_success", "sequence_success",
        "goal_definition_relaxed_success", "goal_definition_relaxed_instance_id",
        "observation_turn_count", "observation_turn_limit", "step_count",
        "decision_trace_count", "applied_action_step_count",
        "episode_step_budget", "no_fresh_action_count", "target_distance_m",
        "target_visibility_fraction", "relaxed_to_strict_target_distance_m", "spl",
        "coverage_ratio", "path_ratio", "interaction_success_count", "interaction_correct_count",
        "interaction_attempt_count", "interaction_failure_reasons", "action_timeout_count",
        "tf_warning_count", "actionlib_wait_without_goal_count", "mllm_error_count",
        "output_frame_count", "capture_complete", "encoding_complete", "complete",
        "result_path", "topdown_path", "video_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in episodes:
            writer.writerow({
                "episode_index": row["episode_index"], "group": row["group"], "case_id": row["case_id"],
                "terminal_reason": row["terminal_reason"], "diagnosis": row["diagnosis"],
                "paper_sr_success": row["paper_sr_success"],
                "legacy_interaction_conditioned_success": row["legacy_interaction_conditioned_success"],
                "task_success": row["task_success"], "nav_success": row["nav_success"],
                "required_interaction_success": row["required_interaction_success"],
                "sequence_success": row["sequence_success"],
                "goal_definition_relaxed_success": row["goal_definition_relaxed_success"],
                "goal_definition_relaxed_instance_id": row["goal_definition_relaxed_instance_id"],
                "observation_turn_count": row["observation_turn_count"],
                "observation_turn_limit": row["observation_turn_limit"],
                "step_count": row["step_count"], "decision_trace_count": row["decision_trace_count"],
                "applied_action_step_count": row["applied_action_step_count"],
                "episode_step_budget": row["episode_step_budget"], "no_fresh_action_count": row["no_fresh_action_count"],
                "target_distance_m": row["target_distance_m"], "target_visibility_fraction": row["target_visibility_fraction"],
                "relaxed_to_strict_target_distance_m": row["visual"]["manifest"].get("relaxed_to_strict_target_distance_m"),
                "spl": row["spl"], "coverage_ratio": row["spatial"]["coverage_ratio"],
                "path_ratio": row["spatial"]["path_ratio"],
                "interaction_success_count": row["interaction"]["success_count"],
                "interaction_correct_count": row["interaction"]["correct_count"],
                "interaction_attempt_count": row["interaction"]["attempt_count"],
                "interaction_failure_reasons": json.dumps(row["interaction"]["failure_reasons"], ensure_ascii=False),
                "action_timeout_count": row["logs"]["counts"].get("action_timeout", 0),
                "tf_warning_count": row["logs"]["counts"].get("tf_wait_warning", 0),
                "actionlib_wait_without_goal_count": row["logs"]["counts"].get("actionlib_wait_without_goal", 0),
                "mllm_error_count": row["mllm"]["error_count"],
                "output_frame_count": row["recording"]["output_frame_count"],
                "capture_complete": row["recording"]["capture_complete"],
                "encoding_complete": row["recording"]["encoding_complete"],
                "complete": row["recording"]["complete"],
                "result_path": row["paths"]["result"], "topdown_path": row["visual"]["topdown"],
                "video_path": row["paths"]["video"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path, help="Batch evaluation directory containing summary.json")
    parser.add_argument("--output-dir", type=Path, help="Defaults to EVALUATION_DIR/offline_analysis")
    parser.add_argument("--episode-indices", type=int, nargs="+", help="Analyze a deterministic subset")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--audit-notes",
        type=Path,
        help="Optional structured human visual-audit notes; source_evaluation_dir must match",
    )
    parser.add_argument("--no-images", action="store_true", help="Skip contact-sheet generation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_dir = (args.output_dir or evaluation_dir / "offline_analysis").expanduser().resolve()
    summary_path = evaluation_dir / "summary.json"
    summary = read_json(summary_path)
    if not isinstance(summary, dict) or not isinstance(summary.get("episodes"), list):
        raise ValueError(f"Invalid or missing batch summary: {summary_path}")
    round_contract, round_contract_error = load_round_contract(evaluation_dir)
    operational_rows = {
        int(row["episode_index"]): dict(row) for row in summary["episodes"]
    }
    canonical_rows = (round_contract or {}).get("episodes") or []
    canonical_by_index = {
        int(row["episode_index"]): row for row in canonical_rows
        if row.get("episode_index") is not None
    }
    if canonical_by_index:
        rows = []
        for index, canonical in sorted(canonical_by_index.items()):
            row = dict(operational_rows.get(index, {"episode_index": index}))
            artifacts = canonical.get("artifacts") or {}
            result_path = artifacts.get("episode_result")
            debug_path = artifacts.get("debug_summary")
            if not result_path:
                raise ValueError(f"Episode {index}: canonical round contract has no result")
            row["episode_result_path"] = result_path
            if debug_path:
                row["attempt_dir"] = str(Path(debug_path).parent.parent)
            row["attempt"] = (canonical.get("completion") or {}).get("attempt")
            rows.append(row)
    else:
        rows = list(summary["episodes"])
    if args.episode_indices:
        requested = set(args.episode_indices)
        rows = [row for row in rows if int(row["episode_index"]) in requested]
        missing = requested - {int(row["episode_index"]) for row in rows}
        if missing:
            raise ValueError(f"Requested episodes not present in summary: {sorted(missing)}")
    if not rows:
        raise ValueError("No episodes selected")
    rows = sorted(rows, key=lambda row: int(row["episode_index"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [analyze_episode(evaluation_dir, row) for row in rows]
    for episode in episodes:
        canonical = canonical_by_index.get(episode["episode_index"])
        if not canonical:
            continue
        episode["canonical"] = {
            "completion": canonical.get("completion"),
            "paper_metrics": canonical.get("paper_metrics"),
            "rgb_step_sync": canonical.get("rgb_step_sync"),
        }
        rgb = canonical.get("rgb_step_sync") or {}
        episode["recording"]["raw_step_sync_complete"] = rgb.get("raw_step_sync_complete")
        episode["recording"]["raw_capture_complete"] = rgb.get("step_sync_capture_complete")
        episode["recording"]["expected_step_sync_count"] = rgb.get("expected_step_sync_count")
        episode["recording"]["raw_step_sync_count"] = rgb.get("step_sync_count")
        if rgb.get("raw_step_sync_complete") is not None or rgb.get("step_sync_capture_complete") is not None:
            episode["recording"]["capture_complete"] = bool(
                rgb.get("raw_step_sync_complete") and rgb.get("step_sync_capture_complete")
            )
            episode["recording"]["complete"] = bool(
                episode["recording"]["capture_complete"]
                and episode["recording"]["encoding_complete"]
                and episode["recording"]["summary_present"]
                and episode["recording"]["video_present"]
            )
    audit_notes = apply_audit_notes(episodes, evaluation_dir, args.audit_notes)
    if not args.no_images:
        build_visuals(episodes, output_dir)
    baseline = None if args.no_baseline else read_json(args.baseline, None)
    aggregate_data = aggregate(episodes, summary, round_contract)
    raw_complete = sum(
        row["recording"].get("raw_step_sync_complete") is True
        and row["recording"].get("raw_capture_complete") is True
        for row in episodes
    )
    shortfalls = []
    for row in episodes:
        expected = row["recording"].get("expected_step_sync_count")
        observed = row["recording"].get("raw_step_sync_count")
        if expected is not None and observed is not None and int(observed) < int(expected):
            shortfalls.append({
                "episode_index": row["episode_index"],
                "expected": int(expected),
                "observed": int(observed),
                "shortfall": int(expected) - int(observed),
            })
    aggregate_data["recording"]["raw_capture_complete_episode_count"] = raw_complete
    aggregate_data["recording"]["raw_capture_shortfalls"] = shortfalls
    provenance_inputs = {
        "summary": file_fingerprint(summary_path),
        "batch_manifest": file_fingerprint(evaluation_dir / "batch_manifest.json"),
        "launch_config": file_fingerprint(evaluation_dir / "launch_config.json"),
        "baseline": file_fingerprint(args.baseline),
    }
    if args.audit_notes:
        provenance_inputs["visual_audit_notes"] = file_fingerprint(args.audit_notes)
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "evaluation_dir": str(evaluation_dir),
        "output_dir": str(output_dir),
        "source_summary": str(summary_path),
        "provenance": {
            "invocation_argv": list(sys.argv),
            "python_version": sys.version,
            "analyzer": file_fingerprint(Path(__file__)),
            "inputs": provenance_inputs,
        },
        "canonical_round_summary": {
            "available": round_contract is not None,
            "error": round_contract_error,
            "completion": ((round_contract or {}).get("round_diagnostics") or {}).get("completion"),
            "paper_metrics": (round_contract or {}).get("paper_metrics"),
            "round_diagnostics": (round_contract or {}).get("round_diagnostics"),
            "warnings": (round_contract or {}).get("warnings") or [],
        },
        "resource_telemetry": resource_telemetry_summary(evaluation_dir / "resource_telemetry.csv"),
        "audit_notes": {
            "path": str(args.audit_notes.resolve()) if args.audit_notes else None,
            "loaded": audit_notes is not None,
        },
        "aggregate": aggregate_data,
        "episodes": episodes,
    }
    analysis["offline_reproduction_checks"] = build_reproduction_checks(episodes, aggregate_data)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / "episodes.csv", episodes)
    (output_dir / "evidence_index.json").write_text(
        json.dumps(evidence_index(episodes), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "offline_reproduction_checks.json").write_text(
        json.dumps(analysis["offline_reproduction_checks"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_report.md").write_text(
        render_markdown(analysis, baseline), encoding="utf-8"
    )
    (output_dir / "analysis_report_zh.md").write_text(
        render_markdown_zh(analysis, baseline), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "episodes": len(episodes),
        "terminal_reason_counts": analysis["aggregate"]["terminal_reason_counts"],
        "diagnosis_counts": analysis["aggregate"]["diagnosis_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
