from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import benchmark_door_state_scan as door_scan
from scripts.InteractiveNav import build_container_interaction_benchmark as container_builder
from scripts.InteractiveNav import build_door_interaction_benchmark as door_builder
from scripts.InteractiveNav import container_scene_probe as probe
from scripts.InteractiveNav import explore_molmo_interactions as emi


DEFAULT_ROUGH_CATALOG = REPO_ROOT / (
    "scripts/InteractiveNav/output/container_interaction_all_houses/rough_catalog/"
    "rough_catalog.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1"
)
SELECTION_SCOPE = "all_container_rough_houses_with_strict_pairs"
CANDIDATE_SELECTION = "all_open_gt_path_crosses_interactive_door_portal"
MIXED_REQUIRED_ROLE = "verified_subset_annotation_not_rough_input_gate"
DOOR_APPROACH_SEMANTICS = "geometric_path_backoff_hint_not_manipulation_validated"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(probe.to_jsonable(payload), indent=2, ensure_ascii=False) + "\n")
    temporary.chmod(0o644)
    temporary.replace(path)


def load_rough_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "container_rough_catalog_v1":
        raise ValueError("Mixed rough production only accepts container_rough_catalog_v1")
    return payload


def load_episodes(benchmark_dir: Path) -> list[dict[str, Any]]:
    return container_builder.load_benchmark_episodes(benchmark_dir)


def strict_pair_count(house: dict[str, Any]) -> int:
    return sum(
        len(container.get("strict_contained_objects", []))
        for container in house.get("containers", [])
    )


def house_ids(
    catalog: dict[str, Any],
    *,
    explicit: str | None,
    max_houses: int | None,
) -> list[int]:
    available = []
    for row in catalog.get("houses", []):
        nested_pair_count = strict_pair_count(row)
        declared_pair_count = int(row.get("strict_pair_count", nested_pair_count))
        if declared_pair_count != nested_pair_count:
            raise ValueError(
                f"House {row.get('house_index')} strict pair count mismatch: "
                f"declared={declared_pair_count}, nested={nested_pair_count}"
            )
        if nested_pair_count > 0:
            available.append(int(row["house_index"]))
    if explicit:
        requested = [int(value) for value in explicit.split(",") if value.strip()]
        available_set = set(available)
        missing = [value for value in requested if value not in available_set]
        if missing:
            raise ValueError(f"Requested houses are absent from the container rough catalog: {missing}")
        available = requested
    if max_houses is not None:
        available = available[:max_houses]
    return available


def bin_counts(values: list[float], edges: list[float]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        placed = False
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            if low <= value < high:
                counts[f"[{low:g},{high:g})"] += 1
                placed = True
                break
        if not placed:
            counts[f"[{edges[-1]:g},inf)"] += 1
    return dict(counts)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def closed_path_evidence(
    open_path: np.ndarray,
    closed_path: np.ndarray | None,
    *,
    min_shortcut_delta_m: float,
    min_shortcut_ratio: float,
) -> dict[str, Any]:
    open_length = emi.path_length(open_path)
    if open_length is None:
        raise ValueError("Open path must contain at least two points")
    closed_length = emi.path_length(closed_path)
    delta = None if closed_length is None else float(closed_length - open_length)
    ratio = None if delta is None else float(delta / max(open_length, 1e-6))
    shortcut_verified = bool(
        delta is not None
        and delta >= float(min_shortcut_delta_m)
        and ratio is not None
        and ratio >= float(min_shortcut_ratio)
    )
    return {
        "closed_path_found": closed_path is not None,
        "all_open_path_length_m": float(open_length),
        "closed_path_length_m": closed_length,
        "path_length_delta_m": delta,
        "path_length_ratio_delta": ratio,
        "shortcut_thresholds": {
            "min_delta_m": float(min_shortcut_delta_m),
            "min_ratio": float(min_shortcut_ratio),
        },
        "mixed_shortcut_verified": shortcut_verified,
    }


def classify_non_crossing_pair(path_status_counts: Counter[str]) -> str:
    if path_status_counts["source_episode_available"] == 0:
        return "no_available_source_episode"
    if path_status_counts["open_path_found"] == 0:
        return "no_open_path_from_any_source"
    if path_status_counts["door_crossing_path_found"] == 0:
        return "open_paths_without_interactive_door"
    raise ValueError("A pair with a door-crossing path must be emitted as a rough candidate")


def summarize_candidates(
    candidates: list[dict[str, Any]],
    houses: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    elapsed_sec: float,
    *,
    expected_house_count: int | None = None,
    expected_pair_count: int | None = None,
) -> dict[str, Any]:
    path_lengths = [float(row["all_open_path_length_m"]) for row in candidates]
    total_interactions = [int(row["estimated_total_interaction_count"]) for row in candidates]
    crossing_house_count = len(
        {int(row["house_index"]) for row in candidates}
    )
    required_candidates = [
        row for row in candidates if bool(row.get("mixed_required_verified", False))
    ]
    set_required_candidates = [
        row
        for row in candidates
        if bool(row.get("mixed_door_set_required_verified", False))
    ]
    shortcut_candidates = [
        row for row in candidates if bool(row.get("mixed_shortcut_verified", False))
    ]
    set_shortcut_candidates = [
        row
        for row in candidates
        if bool(row.get("mixed_door_set_shortcut_verified", False))
    ]
    crossing_only_candidates = [
        row
        for row in candidates
        if row.get("rough_candidate_type") == "door_crossing_only"
    ]
    mixed_required_house_count = len(
        {int(row["house_index"]) for row in required_candidates}
    )
    required_path_lengths = [
        float(row["selected_required_evidence"]["all_open_path_length_m"])
        for row in required_candidates
        if row.get("selected_required_evidence") is not None
    ]
    shortcut_deltas = []
    for row in shortcut_candidates:
        evidence = row.get("selected_shortcut_evidence") or {}
        selected_root = evidence.get("selected_shortcut_door_root")
        checks = evidence.get("door_requirement_checks", [])
        selected_check = next(
            (
                check
                for check in checks
                if check.get("door_root_name") == selected_root
            ),
            None,
        )
        if selected_check and selected_check.get("path_length_delta_m") is not None:
            shortcut_deltas.append(float(selected_check["path_length_delta_m"]))
        elif (
            (evidence.get("all_crossed_doors_closed_evidence") or {}).get(
                "path_length_delta_m"
            )
            is not None
        ):
            shortcut_deltas.append(
                float(
                    evidence["all_crossed_doors_closed_evidence"][
                        "path_length_delta_m"
                    ]
                )
            )
    set_shortcut_deltas = [
        float(evidence["path_length_delta_m"])
        for row in set_shortcut_candidates
        for evidence in [
            (row.get("selected_set_shortcut_evidence") or {}).get(
                "all_crossed_doors_closed_evidence"
            )
            or {}
        ]
        if evidence.get("path_length_delta_m") is not None
    ]
    scanned_pair_count = sum(
        int(row.get("source_container_pair_count", 0)) for row in houses
    )
    resolved_pair_count = sum(
        int(row.get("resolved_strict_pair_count", row.get("source_container_pair_count", 0)))
        for row in houses
    )
    expected_house_count = (
        len(houses) if expected_house_count is None else int(expected_house_count)
    )
    expected_pair_count = (
        scanned_pair_count if expected_pair_count is None else int(expected_pair_count)
    )
    return {
        "schema_version": "mixed_rough_catalog_summary_v1",
        "selection_scope": SELECTION_SCOPE,
        "candidate_selection": CANDIDATE_SELECTION,
        "mixed_required_role": MIXED_REQUIRED_ROLE,
        "door_required_house_prefilter_used": False,
        "door_approach_semantics": DOOR_APPROACH_SEMANTICS,
        "expected_strict_house_count": expected_house_count,
        "expected_strict_pair_count": expected_pair_count,
        "completed_house_count": len(houses),
        "source_container_pair_count": scanned_pair_count,
        "resolved_strict_pair_count": resolved_pair_count,
        "pair_coverage_complete": bool(
            not failures
            and len(houses) == expected_house_count
            and scanned_pair_count == expected_pair_count
            and resolved_pair_count == expected_pair_count
        ),
        "door_crossing_pair_count": len(candidates),
        "door_crossing_pair_rate": (
            float(len(candidates) / resolved_pair_count) if resolved_pair_count else 0.0
        ),
        "door_crossing_house_count": crossing_house_count,
        "door_crossing_house_rate": (
            float(crossing_house_count / len(houses)) if houses else 0.0
        ),
        "no_door_crossing_house_count": len(houses) - crossing_house_count,
        "mixed_required_pair_count": len(required_candidates),
        "mixed_required_pair_rate": (
            float(len(required_candidates) / resolved_pair_count)
            if resolved_pair_count
            else 0.0
        ),
        "mixed_required_within_crossing_rate": (
            float(len(required_candidates) / len(candidates)) if candidates else 0.0
        ),
        "mixed_required_house_count": mixed_required_house_count,
        "mixed_required_house_rate": (
            float(mixed_required_house_count / len(houses)) if houses else 0.0
        ),
        "no_mixed_required_house_count": len(houses) - mixed_required_house_count,
        "mixed_door_set_required_pair_count": len(set_required_candidates),
        "mixed_shortcut_pair_count": len(shortcut_candidates),
        "mixed_door_set_shortcut_pair_count": len(set_shortcut_candidates),
        "mixed_shortcut_within_crossing_rate": (
            float(len(shortcut_candidates) / len(candidates)) if candidates else 0.0
        ),
        "door_crossing_only_pair_count": len(crossing_only_candidates),
        "failure_count": len(failures),
        "elapsed_sec": float(elapsed_sec),
        "path_length_m": {
            **numeric_summary(path_lengths),
            "bins": bin_counts(path_lengths, [0.0, 3.0, 5.0, 8.0, 12.0, 20.0]),
        },
        "mixed_required_path_length_m": {
            **numeric_summary(required_path_lengths),
            "bins": bin_counts(
                required_path_lengths, [0.0, 3.0, 5.0, 8.0, 12.0, 20.0]
            ),
        },
        "mixed_shortcut_delta_m": {
            **numeric_summary(shortcut_deltas),
            "bins": bin_counts(shortcut_deltas, [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]),
        },
        "mixed_door_set_shortcut_delta_m": {
            **numeric_summary(set_shortcut_deltas),
            "bins": bin_counts(
                set_shortcut_deltas, [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
            ),
        },
        "rough_candidate_type_counts": dict(
            Counter(str(row["rough_candidate_type"]) for row in candidates)
        ),
        "container_category_counts": dict(
            Counter(str(row["container_category"]) for row in candidates)
        ),
        "target_category_counts": dict(
            Counter(str(row["target_category"]) for row in candidates)
        ),
        "controlling_joint_type_counts": dict(
            Counter(str(row["controlling_joint_type"]) for row in candidates)
        ),
        "estimated_interaction_type_counts": dict(
            Counter(
                interaction_type
                for row in candidates
                for interaction_type in row.get("estimated_interaction_types", [])
            )
        ),
        "container_interaction_count_distribution": dict(
            Counter(str(row["estimated_container_interaction_count"]) for row in candidates)
        ),
        "channel_interaction_count_distribution": dict(
            Counter(str(row["estimated_channel_interaction_count"]) for row in candidates)
        ),
        "total_interaction_count": {
            **numeric_summary([float(value) for value in total_interactions]),
            "distribution": dict(Counter(str(value) for value in total_interactions)),
        },
        "crossed_door_root_count_distribution": dict(
            Counter(str(len(row["crossed_door_roots"])) for row in candidates)
        ),
        "required_door_root_count_distribution": dict(
            Counter(
                str(len(row.get("selected_required_evidence", {}).get("required_door_roots", [])))
                for row in required_candidates
            )
        ),
        "pair_result_counts": dict(
            sum(
                (
                    Counter(
                        {
                            reason: int(count)
                            for reason, count in house.get("pair_result_counts", {}).items()
                        }
                    )
                    for house in houses
                ),
                Counter(),
            )
        ),
        "source_path_status_counts": dict(
            sum(
                (
                    Counter(
                        {
                            reason: int(count)
                            for reason, count in house.get(
                                "source_path_status_counts", {}
                            ).items()
                        }
                    )
                    for house in houses
                ),
                Counter(),
            )
        ),
        "door_requirement_status_counts": dict(
            sum(
                (
                    Counter(
                        {
                            reason: int(count)
                            for reason, count in house.get(
                                "door_requirement_status_counts", {}
                            ).items()
                        }
                    )
                    for house in houses
                ),
                Counter(),
            )
        ),
        "rejection_reason_counts": dict(
            Counter(
                reason
                for house in houses
                for reason, count in house.get("rejection_reason_counts", {}).items()
                for _ in range(int(count))
            )
        ),
    }


def public_door_record(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        "name": record["name"],
        "category": record.get("category"),
        "hinge_body_names": list(
            record.get("hinge_body_names", record.get("children", []))
        ),
        "aabb_center": np.asarray(record["aabb_center"], dtype=float).tolist(),
        "aabb_size": np.asarray(record["aabb_size"], dtype=float).tolist(),
    }
    for key in (
        "portal_frame_body_name",
        "portal_half_width_m",
        "portal_half_thickness_m",
    ):
        if key in record:
            result[key] = record[key]
    for key in ("portal_center_xy", "portal_tangent_xy", "portal_normal_xy"):
        if key in record:
            result[key] = np.asarray(record[key], dtype=float).tolist()
    return result


def path_door_approach(
    path: np.ndarray,
    door_record: dict[str, Any],
    *,
    padding_m: float,
    sample_step_m: float,
    standoff_m: float,
) -> dict[str, Any] | None:
    dense = door_scan.densify_polyline(np.asarray(path, dtype=float), sample_step_m)
    if len(dense) == 0:
        return None
    crossing = door_scan.path_door_crossing_details(
        path,
        door_record,
        padding_m=padding_m,
        sample_step_m=sample_step_m,
    )
    if not crossing["traverses"]:
        return None
    entry_index = int(crossing["entry_index"])
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_distance = max(0.0, float(cumulative[entry_index]) - standoff_m)
    approach_index = int(np.searchsorted(cumulative, target_distance, side="right") - 1)
    approach_index = min(max(approach_index, 0), entry_index)
    approach_xy = dense[approach_index]
    return {
        "door_entry_index": entry_index,
        "door_entry_distance_m": float(cumulative[entry_index]),
        "approach_path_index": approach_index,
        "approach_xy": np.asarray(approach_xy, dtype=float).tolist(),
        "approach_distance_from_start_m": float(cumulative[approach_index]),
        "standoff_m": float(cumulative[entry_index] - cumulative[approach_index]),
    }


def rough_joint_candidates(
    args: argparse.Namespace,
    ctx: probe.LoadedContext,
    container: dict[str, Any],
    object_record: dict[str, Any],
    dependencies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    joints_by_index = {int(row["joint_index"]): row for row in container["joints"]}
    for joint in container["joints"]:
        joint_index = int(joint["joint_index"])
        joint_type = probe.joint_mujoco_type_name(ctx.env, joint)
        if joint_type not in {"hinge", "slide"}:
            continue
        try:
            joint_sequence = probe.articulation_dependency_order(joint_index, dependencies)
        except ValueError:
            continue
        containment = None
        if joint_type == "slide":
            containment = probe.object_in_closed_joint_box(
                ctx,
                container,
                joint,
                object_record,
                padding=args.drawer_box_padding,
            )
            if not containment.get("contained", False):
                continue
        candidates.append(
            {
                "joint_index": joint_index,
                "joint_name": joint["joint_name"],
                "joint_type": joint_type,
                "joint_sequence": joint_sequence,
                "joint_sequence_types": [
                    probe.joint_mujoco_type_name(ctx.env, joints_by_index[index])
                    for index in joint_sequence
                ],
                "containment": containment,
            }
        )
    candidates.sort(
        key=lambda row: (
            len(row["joint_sequence"]),
            int(row["joint_index"]),
        )
    )
    return candidates


def scan_house(
    args: argparse.Namespace,
    house_catalog: dict[str, Any],
    episodes_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    house_index = int(house_catalog["house_index"])
    started_at = time.perf_counter()
    source_episode_indices = [int(value) for value in house_catalog["source_episode_indices"]]
    template_episode = episodes_by_index[int(house_catalog["template_episode_index"])]
    context = None
    rejections: Counter[str] = Counter()
    pair_result_counts: Counter[str] = Counter()
    source_path_status_counts: Counter[str] = Counter()
    door_requirement_status_counts: Counter[str] = Counter()
    output_candidates: list[dict[str, Any]] = []
    source_pair_count = strict_pair_count(house_catalog)
    declared_pair_count = int(
        house_catalog.get("strict_pair_count", source_pair_count)
    )
    if declared_pair_count != source_pair_count:
        raise ValueError(
            f"House {house_index} strict pair count mismatch: "
            f"declared={declared_pair_count}, nested={source_pair_count}"
        )
    try:
        context = container_builder.load_episode_context(args, template_episode)
        container_builder.open_all_available_doors(context)
        _, initial_containers = probe.collect_scene_records(context)
        container_builder.close_all_containers(context.env, initial_containers)
        records, containers = probe.collect_scene_records(context)
        containers_by_name = {row["name"]: row for row in containers}
        objects_by_name = {row["name"]: row for row in records if probe.is_target_like(row)}
        open_map, doorway_analysis = emi.build_live_procthor_map(
            context.env.current_model,
            context.env.current_data,
            model_path=str(context.env.current_model_path),
            px_per_m=args.px_per_m,
            agent_radius=context.cfg.task_sampler_config.robot_safety_radius,
            open_threshold=args.open_threshold,
            treat_all_non_interactive_doorways_as_open=True,
            return_doorway_analysis=True,
        )
        live_door_records = emi.collect_interactive_door_root_object_records(
            context.env, doorway_analysis
        )
        doors_by_name = {row["name"]: row for row in live_door_records}
        closed_map_cache: dict[tuple[str, ...], Any] = {}
        closed_transition_cache: dict[str, dict[str, Any]] = {}
        closed_map_failures: set[tuple[str, ...]] = set()

        def closed_roots_map(root_names: list[str] | tuple[str, ...]):
            key = tuple(sorted(set(root_names)))
            if not key:
                return open_map
            if key in closed_map_failures:
                return None
            if key not in closed_map_cache:
                try:
                    container_builder.open_all_available_doors(context)
                    for root_name in key:
                        transition = emi.set_door_root_state(
                            context.env, doorway_analysis, root_name, "closed"
                        )
                        if len(key) == 1:
                            closed_transition_cache[root_name] = transition
                    closed_map_cache[key] = emi.build_live_procthor_map(
                        context.env.current_model,
                        context.env.current_data,
                        model_path=str(context.env.current_model_path),
                        px_per_m=args.px_per_m,
                        agent_radius=context.cfg.task_sampler_config.robot_safety_radius,
                        open_threshold=args.open_threshold,
                        treat_all_non_interactive_doorways_as_open=True,
                    )
                except Exception:
                    closed_map_failures.add(key)
                    container_builder.open_all_available_doors(context)
                    return None
                finally:
                    container_builder.open_all_available_doors(context)
            return closed_map_cache[key]

        def closed_map(root_name: str):
            return closed_roots_map([root_name])

        dependencies_cache: dict[str, list[dict[str, Any]]] = {}
        for rough_container in house_catalog.get("containers", []):
            strict_objects = rough_container.get("strict_contained_objects", [])
            if not strict_objects:
                continue
            container = containers_by_name.get(rough_container["name"])
            if container is None:
                rejections["container_missing_in_live_scene"] += len(strict_objects)
                pair_result_counts["container_missing_in_live_scene"] += len(
                    strict_objects
                )
                continue
            if container["name"] not in dependencies_cache:
                dependencies_cache[container["name"]] = probe.infer_joint_open_dependencies(
                    context.env, container, method="front_occlusion"
                )
            dependencies = dependencies_cache[container["name"]]
            rough_goal_xy = emi.nearest_free_point_xy(
                open_map,
                np.asarray(container["aabb_center"], dtype=float)[:2],
                max_radius_px=args.max_goal_search_radius_px,
            )
            if rough_goal_xy is None:
                rejections["no_nearest_free_container_goal"] += len(strict_objects)
                pair_result_counts["no_nearest_free_container_goal"] += len(
                    strict_objects
                )
                continue
            for rough_object in strict_objects:
                object_record = objects_by_name.get(rough_object["name"])
                if object_record is None:
                    rejections["object_missing_in_live_scene"] += 1
                    pair_result_counts["object_missing_in_live_scene"] += 1
                    continue
                if not probe.compute_relation(container, object_record)["inside_aabb"]:
                    rejections["pair_not_inside_live_aabb"] += 1
                    pair_result_counts["pair_not_inside_live_aabb"] += 1
                    continue
                joint_candidates = rough_joint_candidates(
                    args, context, container, object_record, dependencies
                )
                if not joint_candidates:
                    rejections["no_rough_controlling_joint"] += 1
                    pair_result_counts["no_rough_controlling_joint"] += 1
                    continue

                path_options = []
                path_status_counts: Counter[str] = Counter()
                requirement_status_counts: Counter[str] = Counter()
                selected_joint = joint_candidates[0]
                for source in rough_object.get("source_starts", []):
                    path_status_counts["source_start_declared"] += 1
                    source_index = int(source["episode_index"])
                    if source_index not in episodes_by_index:
                        path_status_counts["source_episode_missing"] += 1
                        continue
                    path_status_counts["source_episode_available"] += 1
                    start_xy = np.asarray(source["robot_base_pose"][:2], dtype=float)
                    open_path = emi.compute_path_from_map(
                        open_map, start_xy, rough_goal_xy, downscale_factor=1
                    )
                    if open_path is None:
                        path_status_counts["open_path_not_found"] += 1
                        continue
                    path_status_counts["open_path_found"] += 1
                    crossed = door_scan.traversed_interactive_doors_on_path(
                        context.env,
                        doorway_analysis,
                        open_path,
                        padding_m=args.door_on_path_padding_m,
                        sample_step_m=args.path_region_sample_step_m,
                    )
                    crossed = door_builder.sort_door_records_by_path_entry(
                        crossed, open_path, args
                    )
                    ignored_initial_exit_door_roots = []
                    for live_door_record in live_door_records:
                        crossing_details = door_scan.path_door_crossing_details(
                            open_path,
                            live_door_record,
                            padding_m=args.door_on_path_padding_m,
                            sample_step_m=args.path_region_sample_step_m,
                        )
                        if crossing_details["ignored_initial_region"]:
                            ignored_initial_exit_door_roots.append(
                                live_door_record["name"]
                            )
                    path_status_counts[
                        "start_inside_door_initial_exit_ignored"
                    ] += len(ignored_initial_exit_door_roots)
                    if not crossed:
                        path_status_counts["open_path_without_interactive_door"] += 1
                        continue
                    path_status_counts["door_crossing_path_found"] += 1
                    required = []
                    shortcut_roots = []
                    approach_by_root: dict[str, dict[str, Any]] = {}
                    requirement_checks = []
                    for door_record in crossed:
                        root_name = door_record["name"]
                        approach = path_door_approach(
                            open_path,
                            door_record,
                            padding_m=args.door_on_path_padding_m,
                            sample_step_m=args.path_region_sample_step_m,
                            standoff_m=args.door_approach_standoff_m,
                        )
                        if approach is None:
                            status = "approach_hint_missing"
                            requirement_status_counts[status] += 1
                            requirement_checks.append(
                                {
                                    "door_root_name": root_name,
                                    "status": status,
                                    "closed_path_found": None,
                                    "approach_path_found": None,
                                    "approach": None,
                                }
                            )
                            continue
                        approach_by_root[root_name] = approach
                        single_closed_map = closed_map(root_name)
                        if single_closed_map is None:
                            status = "closed_map_failed"
                            requirement_status_counts[status] += 1
                            requirement_checks.append(
                                {
                                    "door_root_name": root_name,
                                    "status": status,
                                    "closed_path_found": None,
                                    "approach_path_found": None,
                                    "approach": approach,
                                }
                            )
                            continue
                        closed_path = emi.compute_path_from_map(
                            single_closed_map, start_xy, rough_goal_xy, downscale_factor=1
                        )
                        path_evidence = closed_path_evidence(
                            open_path,
                            closed_path,
                            min_shortcut_delta_m=args.min_shortcut_delta_m,
                            min_shortcut_ratio=args.min_shortcut_ratio,
                        )
                        approach_path = emi.compute_path_from_map(
                            single_closed_map,
                            start_xy,
                            np.asarray(approach["approach_xy"], dtype=float),
                            downscale_factor=1,
                        )
                        if closed_path is None and approach_path is not None:
                            status = "mixed_required_verified"
                            required.append(root_name)
                            approach_by_root[root_name] = {
                                **approach,
                                "approach_path_length_m": emi.path_length(approach_path),
                            }
                        elif closed_path is None:
                            status = "goal_blocked_but_approach_unreachable"
                        else:
                            status = "goal_still_reachable_when_closed"
                            if path_evidence["mixed_shortcut_verified"]:
                                status = "mixed_shortcut_verified"
                                shortcut_roots.append(root_name)
                        requirement_status_counts[status] += 1
                        requirement_checks.append(
                            {
                                "door_root_name": root_name,
                                "status": status,
                                **path_evidence,
                                "approach_path_found": approach_path is not None,
                                "approach": approach_by_root[root_name],
                            }
                        )

                    crossed_root_names = [row["name"] for row in crossed]
                    all_crossed_closed_map = closed_roots_map(crossed_root_names)
                    all_crossed_closed_path = (
                        None
                        if all_crossed_closed_map is None
                        else emi.compute_path_from_map(
                            all_crossed_closed_map,
                            start_xy,
                            rough_goal_xy,
                            downscale_factor=1,
                        )
                    )
                    all_crossed_closed_evidence = (
                        None
                        if all_crossed_closed_map is None
                        else closed_path_evidence(
                            open_path,
                            all_crossed_closed_path,
                            min_shortcut_delta_m=args.min_shortcut_delta_m,
                            min_shortcut_ratio=args.min_shortcut_ratio,
                        )
                    )
                    jointly_accessible_approach_roots = []
                    if (
                        all_crossed_closed_map is not None
                        and all_crossed_closed_path is None
                    ):
                        for root_name in crossed_root_names:
                            approach = approach_by_root.get(root_name)
                            if approach is None:
                                continue
                            approach_path = emi.compute_path_from_map(
                                all_crossed_closed_map,
                                start_xy,
                                np.asarray(approach["approach_xy"], dtype=float),
                                downscale_factor=1,
                            )
                            if approach_path is not None:
                                jointly_accessible_approach_roots.append(root_name)
                    mixed_door_set_required_verified = bool(
                        not required
                        and all_crossed_closed_evidence is not None
                        and not all_crossed_closed_evidence["closed_path_found"]
                        and jointly_accessible_approach_roots
                    )
                    mixed_shortcut_verified = bool(
                        not required
                        and not mixed_door_set_required_verified
                        and shortcut_roots
                    )
                    mixed_door_set_shortcut_verified = bool(
                        not required
                        and not mixed_door_set_required_verified
                        and not shortcut_roots
                        and all_crossed_closed_evidence is not None
                        and all_crossed_closed_evidence[
                            "mixed_shortcut_verified"
                        ]
                    )

                    all_scene_closed_evidence = None
                    if args.verify_all_scene_doors_closed:
                        all_scene_closed_map = closed_roots_map(
                            sorted(doors_by_name.keys())
                        )
                        if all_scene_closed_map is not None:
                            all_scene_closed_path = emi.compute_path_from_map(
                                all_scene_closed_map,
                                start_xy,
                                rough_goal_xy,
                                downscale_factor=1,
                            )
                            all_scene_closed_evidence = closed_path_evidence(
                                open_path,
                                all_scene_closed_path,
                                min_shortcut_delta_m=args.min_shortcut_delta_m,
                                min_shortcut_ratio=args.min_shortcut_ratio,
                            )
                    if required:
                        path_status_counts["mixed_required_path_verified"] += 1
                    elif mixed_door_set_required_verified:
                        path_status_counts[
                            "mixed_door_set_required_path_verified"
                        ] += 1
                    elif mixed_shortcut_verified:
                        path_status_counts["mixed_shortcut_path_verified"] += 1
                    elif mixed_door_set_shortcut_verified:
                        path_status_counts[
                            "mixed_door_set_shortcut_path_verified"
                        ] += 1
                    else:
                        path_status_counts["door_crossing_only_path"] += 1
                    selected_crossed_root = crossed[0]["name"]
                    selected_required_root = required[0] if required else None
                    selected_set_required_root = (
                        jointly_accessible_approach_roots[0]
                        if mixed_door_set_required_verified
                        else None
                    )
                    shortcut_checks = [
                        row
                        for row in requirement_checks
                        if row.get("mixed_shortcut_verified")
                    ]
                    shortcut_checks.sort(
                        key=lambda row: float(row["path_length_delta_m"]),
                        reverse=True,
                    )
                    selected_shortcut_root = (
                        shortcut_checks[0]["door_root_name"]
                        if shortcut_checks
                        else (
                            selected_crossed_root if mixed_shortcut_verified else None
                        )
                    )
                    selected_mixed_root = (
                        selected_required_root
                        or selected_set_required_root
                        or selected_shortcut_root
                        or selected_crossed_root
                    )
                    path_options.append(
                        {
                            "source_episode_index": source_index,
                            "source_robot_base_pose": source["robot_base_pose"],
                            "controlling_joint_index": selected_joint["joint_index"],
                            "controlling_joint_name": selected_joint["joint_name"],
                            "controlling_joint_type": selected_joint["joint_type"],
                            "joint_sequence": selected_joint["joint_sequence"],
                            "joint_sequence_types": selected_joint[
                                "joint_sequence_types"
                            ],
                            "rough_nav_goal_xy": np.asarray(rough_goal_xy, dtype=float).tolist(),
                            "rough_nav_goal_source": "nearest_free_point_to_container_aabb_center",
                            "all_open_path_length_m": emi.path_length(open_path),
                            "all_open_path_waypoint_count": int(len(open_path)),
                            "crossed_door_roots": crossed_root_names,
                            "ignored_initial_exit_door_roots": (
                                ignored_initial_exit_door_roots
                            ),
                            "required_door_roots": required,
                            "mixed_required_verified": bool(required),
                            "mixed_door_set_required_verified": (
                                mixed_door_set_required_verified
                            ),
                            "jointly_accessible_approach_roots": (
                                jointly_accessible_approach_roots
                            ),
                            "shortcut_door_roots": shortcut_roots,
                            "mixed_shortcut_verified": mixed_shortcut_verified,
                            "mixed_door_set_shortcut_verified": (
                                mixed_door_set_shortcut_verified
                            ),
                            "selected_crossed_door_root": selected_crossed_root,
                            "selected_crossed_door_approach": approach_by_root.get(
                                selected_crossed_root
                            ),
                            "selected_required_door_root": selected_required_root,
                            "selected_set_required_door_root": (
                                selected_set_required_root
                            ),
                            "selected_shortcut_door_root": selected_shortcut_root,
                            "selected_mixed_door_root": selected_mixed_root,
                            "selected_door_approach": (
                                approach_by_root.get(selected_mixed_root)
                            ),
                            "door_requirement_checks": requirement_checks,
                            "all_crossed_doors_closed_evidence": (
                                all_crossed_closed_evidence
                            ),
                            "all_scene_doors_closed_evidence": (
                                all_scene_closed_evidence
                            ),
                            "door_approach_semantics": DOOR_APPROACH_SEMANTICS,
                        }
                    )
                source_path_status_counts.update(path_status_counts)
                door_requirement_status_counts.update(requirement_status_counts)
                if not path_options:
                    rejection_reason = classify_non_crossing_pair(path_status_counts)
                    rejections[rejection_reason] += 1
                    pair_result_counts[rejection_reason] += 1
                    continue
                path_options.sort(
                    key=lambda row: (
                        float(row["all_open_path_length_m"]),
                        len(row["joint_sequence"]),
                        int(row["source_episode_index"]),
                    )
                )
                required_path_options = [
                    row for row in path_options if row["mixed_required_verified"]
                ]
                set_required_path_options = [
                    row
                    for row in path_options
                    if row["mixed_door_set_required_verified"]
                ]
                shortcut_path_options = [
                    row for row in path_options if row["mixed_shortcut_verified"]
                ]
                set_shortcut_path_options = [
                    row
                    for row in path_options
                    if row["mixed_door_set_shortcut_verified"]
                ]
                selected = path_options[0]
                selected_required_evidence = (
                    required_path_options[0] if required_path_options else None
                )
                selected_set_required_evidence = (
                    set_required_path_options[0]
                    if set_required_path_options
                    else None
                )
                selected_shortcut_evidence = (
                    shortcut_path_options[0] if shortcut_path_options else None
                )
                selected_set_shortcut_evidence = (
                    set_shortcut_path_options[0]
                    if set_shortcut_path_options
                    else None
                )
                mixed_required_verified = selected_required_evidence is not None
                mixed_door_set_required_verified = (
                    selected_set_required_evidence is not None
                )
                mixed_shortcut_verified = selected_shortcut_evidence is not None
                mixed_door_set_shortcut_verified = (
                    selected_set_shortcut_evidence is not None
                )
                if mixed_required_verified:
                    rough_candidate_type = "mixed_required_verified"
                    selected_mixed_evidence = selected_required_evidence
                elif mixed_door_set_required_verified:
                    rough_candidate_type = "mixed_door_set_required_verified"
                    selected_mixed_evidence = selected_set_required_evidence
                elif mixed_shortcut_verified:
                    rough_candidate_type = "mixed_shortcut_verified"
                    selected_mixed_evidence = selected_shortcut_evidence
                elif mixed_door_set_shortcut_verified:
                    rough_candidate_type = "mixed_door_set_shortcut_verified"
                    selected_mixed_evidence = selected_set_shortcut_evidence
                else:
                    rough_candidate_type = "door_crossing_only"
                    selected_mixed_evidence = selected
                pair_result_counts[rough_candidate_type] += 1
                selected_required_root = (
                    selected_required_evidence["selected_required_door_root"]
                    if selected_required_evidence is not None
                    else None
                )
                selected_required_door = (
                    public_door_record(doors_by_name[selected_required_root])
                    if selected_required_root is not None
                    else None
                )
                selected_mixed_root = selected_mixed_evidence.get(
                    "selected_mixed_door_root"
                )
                selected_mixed_door = (
                    public_door_record(doors_by_name[selected_mixed_root])
                    if selected_mixed_root is not None
                    else None
                )
                channel_interaction_count = (
                    len(
                        closed_transition_cache.get(selected_mixed_root, {}).get(
                            "transitions", []
                        )
                    )
                    if selected_mixed_root is not None
                    else 0
                )
                container_interaction_types = [
                    "container_hinged_door"
                    if joint_type == "hinge"
                    else "container_sliding_drawer"
                    for joint_type in selected["joint_sequence_types"]
                ]
                output_candidates.append(
                    {
                        "case_id": (
                            f"mixed_h{house_index}__{probe.sanitize_name(container['name'])}__"
                            f"{probe.sanitize_name(object_record['name'])}"
                        ),
                        "house_index": house_index,
                        "container_name": container["name"],
                        "container_category": str(container.get("category")),
                        "container_asset_id": container.get("asset_id"),
                        "object_name": object_record["name"],
                        "target_category": container_builder.target_category(object_record),
                        "source_episode_indices": source_episode_indices,
                        "path_options": path_options[: args.max_path_options_per_pair],
                        "required_path_options": required_path_options[
                            : args.max_path_options_per_pair
                        ],
                        "set_required_path_options": set_required_path_options[
                            : args.max_path_options_per_pair
                        ],
                        "shortcut_path_options": shortcut_path_options[
                            : args.max_path_options_per_pair
                        ],
                        "set_shortcut_path_options": set_shortcut_path_options[
                            : args.max_path_options_per_pair
                        ],
                        **selected,
                        "rough_candidate_type": rough_candidate_type,
                        "door_crossing_verified": True,
                        "mixed_required_verified": mixed_required_verified,
                        "mixed_door_set_required_verified": (
                            mixed_door_set_required_verified
                        ),
                        "mixed_shortcut_verified": mixed_shortcut_verified,
                        "mixed_door_set_shortcut_verified": (
                            mixed_door_set_shortcut_verified
                        ),
                        "selected_mixed_door_root": selected_mixed_root,
                        "selected_mixed_door": selected_mixed_door,
                        "selected_door_approach": selected_mixed_evidence.get(
                            "selected_door_approach"
                        ),
                        "door_crossing_path_count": path_status_counts[
                            "door_crossing_path_found"
                        ],
                        "mixed_required_path_count": path_status_counts[
                            "mixed_required_path_verified"
                        ],
                        "mixed_door_set_required_path_count": path_status_counts[
                            "mixed_door_set_required_path_verified"
                        ],
                        "mixed_shortcut_path_count": path_status_counts[
                            "mixed_shortcut_path_verified"
                        ],
                        "mixed_door_set_shortcut_path_count": path_status_counts[
                            "mixed_door_set_shortcut_path_verified"
                        ],
                        "source_path_status_counts": dict(path_status_counts),
                        "door_requirement_status_counts": dict(
                            requirement_status_counts
                        ),
                        "selected_required_evidence": selected_required_evidence,
                        "selected_set_required_evidence": (
                            selected_set_required_evidence
                        ),
                        "selected_shortcut_evidence": selected_shortcut_evidence,
                        "selected_set_shortcut_evidence": (
                            selected_set_shortcut_evidence
                        ),
                        "selected_mixed_evidence": selected_mixed_evidence,
                        "selected_required_door": selected_required_door,
                        "estimated_channel_interaction_count": channel_interaction_count,
                        "estimated_container_interaction_count": len(
                            selected["joint_sequence"]
                        ),
                        "estimated_total_interaction_count": channel_interaction_count
                        + len(selected["joint_sequence"]),
                        "estimated_interaction_types": [
                            "channel_hinged_door"
                        ]
                        * channel_interaction_count
                        + container_interaction_types,
                    }
                )
        resolved_pair_count = len(output_candidates) + sum(rejections.values())
        if resolved_pair_count != source_pair_count:
            raise RuntimeError(
                f"House {house_index} pair coverage is incomplete: "
                f"source={source_pair_count}, resolved={resolved_pair_count}"
            )
        if sum(pair_result_counts.values()) != source_pair_count:
            raise RuntimeError(
                f"House {house_index} pair result counts are incomplete: "
                f"source={source_pair_count}, results={sum(pair_result_counts.values())}"
            )
        mixed_required_pair_count = sum(
            bool(row["mixed_required_verified"]) for row in output_candidates
        )
        mixed_set_required_pair_count = sum(
            bool(row["mixed_door_set_required_verified"])
            for row in output_candidates
        )
        mixed_shortcut_pair_count = sum(
            bool(row["mixed_shortcut_verified"]) for row in output_candidates
        )
        mixed_set_shortcut_pair_count = sum(
            bool(row["mixed_door_set_shortcut_verified"])
            for row in output_candidates
        )
        classified_mixed_pair_count = (
            mixed_required_pair_count
            + mixed_set_required_pair_count
            + mixed_shortcut_pair_count
            + mixed_set_shortcut_pair_count
        )
        return (
            {
                "house_index": house_index,
                "source_container_pair_count": source_pair_count,
                "resolved_strict_pair_count": resolved_pair_count,
                "pair_coverage_complete": True,
                "door_crossing_pair_count": len(output_candidates),
                "mixed_required_pair_count": mixed_required_pair_count,
                "mixed_door_set_required_pair_count": (
                    mixed_set_required_pair_count
                ),
                "mixed_shortcut_pair_count": mixed_shortcut_pair_count,
                "mixed_door_set_shortcut_pair_count": (
                    mixed_set_shortcut_pair_count
                ),
                "door_crossing_only_pair_count": (
                    len(output_candidates) - classified_mixed_pair_count
                ),
                "pair_result_counts": dict(pair_result_counts),
                "source_path_status_counts": dict(source_path_status_counts),
                "door_requirement_status_counts": dict(
                    door_requirement_status_counts
                ),
                "rejection_reason_counts": dict(rejections),
                "elapsed_sec": time.perf_counter() - started_at,
            },
            output_candidates,
        )
    finally:
        if context is not None:
            probe.close_context(context)


def scan_worker(args: argparse.Namespace) -> int:
    catalog = load_rough_catalog(args.container_rough_catalog)
    episodes = load_episodes(args.benchmark_dir)
    episodes_by_index = dict(enumerate(episodes))
    requested = set(int(value) for value in args.worker_houses.split(",") if value)
    all_input_houses = [
        row for row in catalog["houses"] if int(row["house_index"]) in requested
    ]
    houses = []
    candidates = []
    failures = []
    completed_house_ids: set[int] = set()
    if args.resume:
        partial_houses_path = args.output_dir / "houses.partial.json"
        partial_candidates_path = args.output_dir / "candidates.partial.json"
        final_catalog_path = args.output_dir / "mixed_rough_catalog.json"
        if partial_houses_path.is_file():
            houses = list(json.loads(partial_houses_path.read_text()))
        elif final_catalog_path.is_file():
            houses = list(json.loads(final_catalog_path.read_text()).get("houses", []))
        completed_house_ids = {
            int(row["house_index"])
            for row in houses
            if int(row["house_index"]) in requested
        }
        houses = [
            row for row in houses if int(row["house_index"]) in completed_house_ids
        ]
        if partial_candidates_path.is_file():
            candidates = list(json.loads(partial_candidates_path.read_text()))
        elif final_catalog_path.is_file():
            candidates = list(
                json.loads(final_catalog_path.read_text()).get("candidates", [])
            )
        candidates = [
            row for row in candidates if int(row["house_index"]) in completed_house_ids
        ]
        if completed_house_ids:
            print(
                f"resume completed_houses={len(completed_house_ids)}/"
                f"{len(all_input_houses)} candidates={len(candidates)}",
                flush=True,
            )
    input_houses = [
        row
        for row in all_input_houses
        if int(row["house_index"]) not in completed_house_ids
    ]
    started_at = time.perf_counter()
    for index, house in enumerate(input_houses, start=len(houses) + 1):
        house_index = int(house["house_index"])
        try:
            house_summary, house_candidates = scan_house(args, house, episodes_by_index)
            houses.append(house_summary)
            candidates.extend(house_candidates)
            print(
                f"[{index}/{len(all_input_houses)}] house={house_index} "
                f"crossing={len(house_candidates)} "
                f"required={house_summary['mixed_required_pair_count']} "
                f"elapsed={house_summary['elapsed_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failures.append({"house_index": house_index, "error": str(exc)})
            print(
                f"[{index}/{len(all_input_houses)}] house={house_index} failed: {exc}",
                flush=True,
            )
        write_json(args.output_dir / "houses.partial.json", houses)
        write_json(args.output_dir / "candidates.partial.json", candidates)
        write_json(args.output_dir / "failures.partial.json", failures)
    expected_pair_count = sum(strict_pair_count(row) for row in all_input_houses)
    summary = summarize_candidates(
        candidates,
        houses,
        failures,
        time.perf_counter() - started_at,
        expected_house_count=len(all_input_houses),
        expected_pair_count=expected_pair_count,
    )
    payload = {
        "schema_version": "mixed_rough_catalog_v1",
        "source_container_rough_catalog": str(args.container_rough_catalog),
        "benchmark_dir": str(args.benchmark_dir),
        "input_scope": {
            "selection_scope": SELECTION_SCOPE,
            "candidate_selection": CANDIDATE_SELECTION,
            "mixed_required_role": MIXED_REQUIRED_ROLE,
            "door_required_house_prefilter_used": False,
            "strict_house_count": len(all_input_houses),
            "strict_pair_count": expected_pair_count,
        },
        "selection_rule": (
            "scan every strict container-object pair in every selected strict-pair house; "
            "keep the pair when an all-open GT path to the rough container interaction goal "
            "crosses from one side of at least one measured interactive-door portal to the other; "
            "ignore only the initial exit for a door whose portal contains the source start; "
            "annotate individual, all-crossed, and all-scene closed-door reachability and classify "
            "required, door-set-required, shortcut-beneficial, and residual crossing-only evidence"
        ),
        "summary": summary,
        "houses": houses,
        "candidates": candidates,
        "failures": failures,
    }
    write_json(args.output_dir / "mixed_rough_catalog.json", payload)
    write_json(args.output_dir / "summary.json", summary)
    return 0 if not failures and summary["pair_coverage_complete"] else 1


def balanced_shards(values: list[int], workers: int) -> list[list[int]]:
    shards = [[] for _ in range(min(max(1, workers), len(values)))]
    for index, value in enumerate(values):
        shards[index % len(shards)].append(value)
    return [row for row in shards if row]


def run_shard(
    args: argparse.Namespace,
    shard_index: int,
    houses: list[int],
    env: dict[str, str],
) -> dict[str, Any]:
    shard_dir = args.output_dir / "shards" / f"shard_{shard_index:03d}"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--container_rough_catalog",
        str(args.container_rough_catalog),
        "--benchmark_dir",
        str(args.benchmark_dir),
        "--output_dir",
        str(shard_dir),
        "--worker_houses",
        ",".join(map(str, houses)),
        "--robot",
        args.robot,
        "--variant",
        args.variant,
        "--seed",
        str(args.seed),
        "--px_per_m",
        str(args.px_per_m),
        "--open_threshold",
        str(args.open_threshold),
        "--max_path_options_per_pair",
        str(args.max_path_options_per_pair),
        "--drawer_box_padding",
        str(args.drawer_box_padding),
        "--door_on_path_padding_m",
        str(args.door_on_path_padding_m),
        "--path_region_sample_step_m",
        str(args.path_region_sample_step_m),
        "--door_approach_standoff_m",
        str(args.door_approach_standoff_m),
        "--min_shortcut_delta_m",
        str(args.min_shortcut_delta_m),
        "--min_shortcut_ratio",
        str(args.min_shortcut_ratio),
        "--max_goal_search_radius_px",
        str(args.max_goal_search_radius_px),
        (
            "--verify_all_scene_doors_closed"
            if args.verify_all_scene_doors_closed
            else "--no-verify_all_scene_doors_closed"
        ),
        "--resume" if args.resume else "--no-resume",
    ]
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = shard_dir / "run.log"
    started_at = time.perf_counter()
    with open(log_path, "a" if args.resume else "w") as handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "shard_index": shard_index,
        "houses": houses,
        "output_dir": str(shard_dir),
        "log_path": str(log_path),
        "returncode": result.returncode,
        "elapsed_sec": time.perf_counter() - started_at,
    }


def merge_shards(
    args: argparse.Namespace,
    requested_houses: list[int],
    shard_results: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    catalog = load_rough_catalog(args.container_rough_catalog)
    requested_set = set(requested_houses)
    requested_catalog_houses = [
        row
        for row in catalog.get("houses", [])
        if int(row["house_index"]) in requested_set
    ]
    expected_pair_count = sum(
        strict_pair_count(row) for row in requested_catalog_houses
    )
    houses = []
    candidates = []
    failures = []
    for result in sorted(shard_results, key=lambda row: row["shard_index"]):
        path = Path(result["output_dir"]) / "mixed_rough_catalog.json"
        if path.exists():
            payload = json.loads(path.read_text())
            houses.extend(payload.get("houses", []))
            candidates.extend(payload.get("candidates", []))
            failures.extend(payload.get("failures", []))
        else:
            failures.append(
                {
                    "houses": result["houses"],
                    "reason": "shard_output_missing",
                    "returncode": result["returncode"],
                    "log_path": result["log_path"],
                }
            )
    order = {house_index: index for index, house_index in enumerate(requested_houses)}
    houses.sort(key=lambda row: order.get(int(row["house_index"]), len(order)))
    candidates.sort(
        key=lambda row: (
            order.get(int(row["house_index"]), len(order)),
            float(row["all_open_path_length_m"]),
            row["case_id"],
        )
    )
    completed = {int(row["house_index"]) for row in houses}
    missing = [value for value in requested_houses if value not in completed]
    summary = summarize_candidates(
        candidates,
        houses,
        failures,
        elapsed_sec,
        expected_house_count=len(requested_houses),
        expected_pair_count=expected_pair_count,
    )
    summary.update(
        {
            "requested_house_count": len(requested_houses),
            "missing_house_count": len(missing),
            "missing_houses": missing,
            "worker_count": args.workers,
        }
    )
    payload = {
        "schema_version": "mixed_rough_catalog_v1",
        "source_container_rough_catalog": str(args.container_rough_catalog),
        "benchmark_dir": str(args.benchmark_dir),
        "input_scope": {
            "selection_scope": SELECTION_SCOPE,
            "candidate_selection": CANDIDATE_SELECTION,
            "mixed_required_role": MIXED_REQUIRED_ROLE,
            "door_required_house_prefilter_used": False,
            "strict_house_count": len(requested_houses),
            "strict_pair_count": expected_pair_count,
        },
        "selection_rule": (
            "scan every strict container-object pair in every selected strict-pair house; "
            "keep the pair when an all-open GT path to the rough container interaction goal "
            "crosses from one side of at least one measured interactive-door portal to the other; "
            "ignore only the initial exit for a door whose portal contains the source start; "
            "annotate individual, all-crossed, and all-scene closed-door reachability and classify "
            "required, door-set-required, shortcut-beneficial, and residual crossing-only evidence"
        ),
        "summary": summary,
        "houses": houses,
        "candidates": candidates,
        "failures": failures,
        "shards": shard_results,
    }
    write_json(args.output_dir / "mixed_rough_catalog.json", payload)
    write_json(args.output_dir / "summary.json", summary)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Second-stage door-crossing rough filtering from container_rough_catalog_v1, "
            "with mixed_required recorded as a verified subset annotation."
        )
    )
    parser.add_argument("--container_rough_catalog", type=Path, default=DEFAULT_ROUGH_CATALOG)
    parser.add_argument("--benchmark_dir", type=Path, default=container_builder.DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_houses", type=int)
    parser.add_argument("--house_indices")
    parser.add_argument("--worker_houses", help=argparse.SUPPRESS)
    parser.add_argument("--robot", default="rby1")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--px_per_m", type=float, default=50.0)
    parser.add_argument("--open_threshold", type=float, default=0.67)
    parser.add_argument("--max_path_options_per_pair", type=int, default=4)
    parser.add_argument("--drawer_box_padding", type=float, default=0.05)
    parser.add_argument("--door_on_path_padding_m", type=float, default=0.2)
    parser.add_argument("--path_region_sample_step_m", type=float, default=0.05)
    parser.add_argument("--door_approach_standoff_m", type=float, default=0.65)
    parser.add_argument("--min_shortcut_delta_m", type=float, default=0.25)
    parser.add_argument("--min_shortcut_ratio", type=float, default=0.02)
    parser.add_argument(
        "--verify_all_scene_doors_closed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also record the door-free reachability/length baseline with every "
            "interactive door root closed."
        ),
    )
    parser.add_argument("--max_goal_search_radius_px", type=int, default=300)
    parser.add_argument("--mujoco_gl", default="egl")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume successful houses from shard partial outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_houses:
        return scan_worker(args)

    catalog = load_rough_catalog(args.container_rough_catalog)
    requested = house_ids(
        catalog, explicit=args.house_indices, max_houses=args.max_houses
    )
    if not requested:
        raise ValueError("No eligible houses were selected")
    requested_set = set(requested)
    requested_catalog_houses = [
        row
        for row in catalog.get("houses", [])
        if int(row["house_index"]) in requested_set
    ]
    expected_pair_count = sum(
        strict_pair_count(row) for row in requested_catalog_houses
    )
    print(
        f"selection_scope={SELECTION_SCOPE} candidate_selection={CANDIDATE_SELECTION} "
        f"door_required_house_prefilter_used=false "
        f"strict_houses={len(requested)} strict_pairs={expected_pair_count}",
        flush=True,
    )
    shards = balanced_shards(requested, args.workers)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["MUJOCO_GL"] = args.mujoco_gl
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mixed-rough")
    started_at = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = {
            executor.submit(run_shard, args, index, shard, env): index
            for index, shard in enumerate(shards)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"shard={result['shard_index']} rc={result['returncode']} "
                f"elapsed={result['elapsed_sec']:.1f}s log={result['log_path']}",
                flush=True,
            )
    payload = merge_shards(args, requested, results, time.perf_counter() - started_at)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return (
        0
        if not payload["failures"]
        and not payload["summary"]["missing_houses"]
        and payload["summary"]["pair_coverage_complete"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
