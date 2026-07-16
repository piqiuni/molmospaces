from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def normalized_entropy(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return None
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count > 0
    )
    return entropy / math.log(len(counts))


def distribution(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    values = list(counts.values())
    return {
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "total": total,
        "unique": len(counts),
        "max_share": None if total == 0 else max(values) / total,
        "min_nonzero_share": None if total == 0 else min(values) / total,
        "normalized_entropy": normalized_entropy(counts),
    }


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    benchmark = load_json(args.benchmark_dir / "benchmark.json")
    valid_pairs = load_json(args.benchmark_dir / "valid_pairs.json")
    rejected_pairs = load_json(args.benchmark_dir / "rejected_pairs.json")
    summary = load_json(args.benchmark_dir / "summary.json")

    requested_by_house = {
        int(house["house_index"]): int(
            house.get("target_sample_count", len(house.get("slots", [])))
        )
        for house in manifest.get("houses", [])
    }
    house_counts = Counter(str(row["house_index"]) for row in valid_pairs)
    container_counts = Counter(
        str(row["target"]["container_category"]).lower() for row in valid_pairs
    )
    object_counts = Counter(
        str(row["target"]["category"]).lower() for row in valid_pairs
    )
    asset_counts = Counter()
    distance_bins = Counter()
    candidate_ranks = Counter()
    reveal_modes = Counter()
    interaction_types = Counter()
    interaction_effects = Counter()
    prerequisite_counts = Counter()
    oracle_counts = Counter()
    schema_versions = Counter()
    path_lengths = []
    initial_visibility_violations = []
    path_violations = []
    success_evidence_violations = []
    duplicate_cases = Counter(row["case_id"] for row in valid_pairs)

    rough_asset_by_container = {}
    if args.rough_catalog and args.rough_catalog.exists():
        rough = load_json(args.rough_catalog)
        houses = rough.get("houses", rough) if isinstance(rough, dict) else rough
        for house in houses:
            for container in house["containers"]:
                rough_asset_by_container[
                    (int(house["house_index"]), container["name"])
                ] = str(container.get("asset_id"))

    for row in valid_pairs:
        validation = row["generation_validation"]
        selection = validation.get("candidate_selection", {})
        distance_bins[str(selection.get("selected_start_distance_bin", "unknown"))] += 1
        candidate_ranks[str(selection.get("candidate_rank", 0))] += 1
        reveal_modes[str(validation.get("reveal_mode", "unknown"))] += 1
        schema_versions[str(row.get("schema_version", "missing"))] += 1
        oracle_counts[str(len(row.get("oracle_plans", [])))] += 1
        for interaction in row.get("interactions", []):
            interaction_types[str(interaction["type"])] += 1
            prerequisite_counts[str(len(interaction.get("prerequisites", [])))] += 1
            interaction_effects.update(str(effect) for effect in interaction["effect_types"])
        start = validation.get("navigation_validation", {})
        if start.get("path_length_m") is not None:
            path_lengths.append(float(start["path_length_m"]))
        if not start.get("path_found", False):
            path_violations.append(row["case_id"])
        if float(start.get("start_visibility_fraction", 0.0)) > 0.0 or int(
            start.get("start_visible_pixels", 0)
        ) > 0:
            initial_visibility_violations.append(row["case_id"])
        success_evidence = validation.get("success_evidence", {})
        if not (
            success_evidence.get("status") == "passed"
            and success_evidence.get("expected_task_success") is True
            and success_evidence.get("distance_passed") is True
            and success_evidence.get("visibility_passed") is True
        ):
            success_evidence_violations.append(row["case_id"])
        asset = rough_asset_by_container.get(
            (int(row["house_index"]), row["target"]["container_name"]), "unknown"
        )
        asset_counts[asset] += 1

    rejected_reasons = Counter(str(row["reason"]) for row in rejected_pairs)
    completed_slots = int(summary.get("completed_candidate_slot_count", len(valid_pairs)))
    requested_slot_count = sum(requested_by_house.values())
    complete_houses = sum(
        house_counts[str(house_index)] >= requested
        for house_index, requested in requested_by_house.items()
    )
    requested_houses = len(requested_by_house)
    quality_gates = {
        "benchmark_episode_count_matches_valid_pairs": len(benchmark) == len(valid_pairs),
        "no_duplicate_case_ids": all(count == 1 for count in duplicate_cases.values()),
        "all_valid_paths_found": not path_violations,
        "all_targets_initially_hidden": not initial_visibility_violations,
        "all_schema_versions_are_v3": set(schema_versions) <= {"interactive_nav_v3"},
        "all_success_evidence_passed": not success_evidence_violations,
        "per_house_within_requested_quota": all(
            house_counts[str(house_index)] <= requested
            for house_index, requested in requested_by_house.items()
        ),
        "all_requested_houses_complete": complete_houses == requested_houses,
        "all_slots_completed": completed_slots == requested_slot_count,
    }
    return {
        "schema_version": "container_interaction_quality_report_v1",
        "inputs": {
            "manifest": str(args.manifest),
            "benchmark_dir": str(args.benchmark_dir),
            "rough_catalog": None
            if args.rough_catalog is None
            else str(args.rough_catalog),
        },
        "completeness": {
            "requested_house_count": requested_houses,
            "requested_slot_count": requested_slot_count,
            "completed_slot_count": completed_slots,
            "generated_episode_count": len(benchmark),
            "valid_pair_count": len(valid_pairs),
            "rejected_attempt_count": len(rejected_pairs),
            "houses_with_complete_quota": complete_houses,
            "houses_with_one_valid_sample": sum(count == 1 for count in house_counts.values()),
            "houses_with_zero_valid_samples": requested_houses - len(house_counts),
            "slot_completion_rate": None
            if requested_slot_count == 0
            else completed_slots / requested_slot_count,
        },
        "balance": {
            "container_categories": distribution(container_counts),
            "object_categories": distribution(object_counts),
            "container_assets": distribution(asset_counts),
            "distance_bins": distribution(distance_bins),
            "reveal_modes": distribution(reveal_modes),
            "interaction_types": distribution(interaction_types),
            "interaction_effects": distribution(interaction_effects),
            "prerequisite_counts": distribution(prerequisite_counts),
            "oracle_plan_counts": distribution(oracle_counts),
            "schema_versions": distribution(schema_versions),
            "candidate_ranks": distribution(candidate_ranks),
            "samples_per_house": distribution(house_counts),
        },
        "navigation": {
            "path_length_m": numeric_summary(path_lengths),
            "path_violations": path_violations,
            "initial_visibility_violations": initial_visibility_violations,
            "success_evidence_violations": success_evidence_violations,
        },
        "rejections": distribution(rejected_reasons),
        "quality_gates": quality_gates,
        "all_quality_gates_passed": all(quality_gates.values()),
    }


def markdown(report: dict[str, Any]) -> str:
    completeness = report["completeness"]
    container_balance = report["balance"]["container_categories"]
    object_balance = report["balance"]["object_categories"]
    distance_balance = report["balance"]["distance_bins"]
    lines = [
        "# Container Interaction Dataset Quality Report",
        "",
        "## Assessment",
        "",
        f"- All quality gates passed: `{report['all_quality_gates_passed']}`",
        f"- Container max share: `{container_balance['max_share']}`",
        f"- Object categories: `{object_balance['unique']}`; max share: `{object_balance['max_share']}`",
        f"- Distance-bin normalized entropy: `{distance_balance['normalized_entropy']}`",
        "",
        "## Completeness",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in completeness.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Balance", ""])
    for name, payload in report["balance"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- unique: `{payload['unique']}`",
                f"- max_share: `{payload['max_share']}`",
                f"- normalized_entropy: `{payload['normalized_entropy']}`",
                "",
                "| Value | Count |",
                "|---|---:|",
            ]
        )
        for value, count in payload["counts"].items():
            lines.append(f"| `{value}` | {count} |")
        lines.append("")
    lines.extend(
        [
            "## Navigation",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in report["navigation"]["path_length_m"].items():
        lines.append(f"| `path_length_m.{key}` | {value} |")
    lines.append(
        f"| `path_violations` | {len(report['navigation']['path_violations'])} |"
    )
    lines.append(
        "| `initial_visibility_violations` | "
        f"{len(report['navigation']['initial_visibility_violations'])} |"
    )
    lines.append("")
    lines.extend(["## Quality Gates", "", "| Gate | Passed |", "|---|---|"])
    for gate, passed in report["quality_gates"].items():
        lines.append(f"| `{gate}` | {passed} |")
    lines.extend(
        [
            "",
            "## Rejections",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in report["rejections"]["counts"].items():
        lines.append(f"| `{reason}` | {count} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate collected container benchmark quality.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--rough_catalog", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "quality_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    (args.output_dir / "quality_report.md").write_text(markdown(report))
    print(json.dumps(report["completeness"], indent=2))
    print(json.dumps(report["quality_gates"], indent=2))
    return 0 if report["all_quality_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
