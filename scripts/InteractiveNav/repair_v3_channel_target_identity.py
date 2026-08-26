"""Create a non-destructive V3 Channel target-identity repair artifact.

The affected Channel builder sampled the navigation terminal pose for the
nearest compatible object instance, but persisted the source task's canonical
instance in V3 target metadata.  The companion audit map contains the exact
instance selected by the original sampler and verifies that every frozen
terminal goal is compatible with it.  This tool updates only those target
references; it deliberately does not change paths, doors, initial state, or
oracle navigation waypoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


EXPECTED_DOMAIN = ["channel"]
EXPECTED_FIXED_TARGET_PATHS = {
    "task.pickup_obj_name",
    "task.pickup_obj_candidates[0]",
    "task_relevant_objects[0]",
    "interactive_nav.target.selected_instance",
    "interactive_nav.target.instruction_consistent_candidates[0]",
    "interactive_nav.generation_validation.success_evidence.target_object_name",
}


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _interactive(episode: dict[str, Any]) -> dict[str, Any]:
    value = episode.get("interactive_nav")
    if not isinstance(value, dict):
        raise ValueError("episode has no interactive_nav payload")
    return value


def _find_exact_value_paths(value: Any, expected: str, path: str = "") -> list[str]:
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_find_exact_value_paths(child, expected, child_path))
        return found
    if isinstance(value, list):
        found = []
        for index, child in enumerate(value):
            found.extend(_find_exact_value_paths(child, expected, f"{path}[{index}]"))
        return found
    return [path] if value == expected else []


def _replace_exact_value(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_exact_value(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_exact_value(child, old, new) for child in value]
    return new if value == old else value


def _terminal_goal_xy(episode: dict[str, Any]) -> list[float]:
    interactive = _interactive(episode)
    plan = interactive.get("oracle_plan")
    if not isinstance(plan, dict):
        plans = interactive.get("oracle_plans")
        plan = plans[0] if isinstance(plans, list) and plans else None
    if not isinstance(plan, dict):
        raise ValueError("episode has no oracle plan")
    candidates = [
        step.get("goal_point")
        for step in plan.get("steps", [])
        if step.get("type") == "navigate" and step.get("reason") == "satisfy_nav_to_obj_success"
    ]
    if len(candidates) != 1 or not isinstance(candidates[0], list) or len(candidates[0]) < 2:
        raise ValueError("episode has no unique terminal navigation goal")
    return [float(candidates[0][0]), float(candidates[0][1])]


def _is_observe_target_path(path: str) -> bool:
    return (
        path.startswith("interactive_nav.oracle_plan.steps[")
        or path.startswith("interactive_nav.oracle_plans[")
    ) and path.endswith("].object_name")


def _validate_pre_repair_episode(episode: dict[str, Any], repair: dict[str, Any]) -> None:
    index = repair["benchmark_index"]
    interactive = _interactive(episode)
    if interactive.get("interaction_domains") != EXPECTED_DOMAIN:
        raise ValueError(f"index={index}: expected Channel domain")
    checks = {
        "case_id": interactive.get("case_id"),
        "legacy_case_type": interactive.get("legacy_case_type"),
        "house_index": episode.get("house_index"),
        "parent_benchmark_episode_index": interactive.get("parent_benchmark_episode_index"),
        "selected_instance": interactive.get("target", {}).get("selected_instance"),
    }
    expected = {
        "case_id": repair["case_id"],
        "legacy_case_type": repair["case_type"],
        "house_index": repair["house_index"],
        "parent_benchmark_episode_index": repair["source_parent_episode_index"],
        "selected_instance": repair["old_selected_target"],
    }
    mismatches = {key: (checks[key], expected[key]) for key in checks if checks[key] != expected[key]}
    if mismatches:
        raise ValueError(f"index={index}: repair-map identity mismatch: {mismatches}")

    old_target = repair["old_selected_target"]
    paths = _find_exact_value_paths(episode, old_target)
    fixed_paths = set(paths) - {path for path in paths if _is_observe_target_path(path)}
    observe_paths = [path for path in paths if _is_observe_target_path(path)]
    if fixed_paths != EXPECTED_FIXED_TARGET_PATHS or len(observe_paths) != 2 or len(paths) != 8:
        raise ValueError(
            f"index={index}: unexpected old-target reference layout: {sorted(paths)}"
        )

    goal_xy = _terminal_goal_xy(episode)
    expected_goal_xy = [float(value) for value in repair["terminal_goal_xy"]]
    if any(abs(actual - expected_value) > 1e-8 for actual, expected_value in zip(goal_xy, expected_goal_xy)):
        raise ValueError(f"index={index}: terminal goal differs from audit map")
    if not repair.get("new_terminal_goal_consistent"):
        raise ValueError(f"index={index}: audit map did not certify new terminal target")
    if float(repair["new_terminal_distance_m"]) > float(repair["allowed_terminal_distance_m"]) + 1e-8:
        raise ValueError(f"index={index}: new target does not satisfy recorded terminal geometry")


def _validate_post_repair_episode(episode: dict[str, Any], repair: dict[str, Any]) -> None:
    index = repair["benchmark_index"]
    old_target = repair["old_selected_target"]
    new_target = repair["new_selected_target"]
    if _find_exact_value_paths(episode, old_target):
        raise ValueError(f"index={index}: old target remains after repair")
    interactive = _interactive(episode)
    task = episode.get("task", {})
    target = interactive.get("target", {})
    if task.get("pickup_obj_name") != new_target or task.get("pickup_obj_candidates") != [new_target]:
        raise ValueError(f"index={index}: task target fields were not repaired")
    if target.get("selected_instance") != new_target or target.get("instruction_consistent_candidates") != [new_target]:
        raise ValueError(f"index={index}: V3 target fields were not repaired")
    relevant = episode.get("task_relevant_objects", [])
    if not relevant or relevant[0] != new_target:
        raise ValueError(f"index={index}: task_relevant_objects target was not repaired")
    success = interactive.get("generation_validation", {}).get("success_evidence", {})
    if success.get("target_object_name") != new_target:
        raise ValueError(f"index={index}: success evidence target was not repaired")

    plans = []
    if isinstance(interactive.get("oracle_plan"), dict):
        plans.append(interactive["oracle_plan"])
    plans.extend(plan for plan in interactive.get("oracle_plans", []) if isinstance(plan, dict))
    observe_steps = [
        step
        for plan in plans
        for step in plan.get("steps", [])
        if step.get("type") == "observe_target"
    ]
    if len(observe_steps) != 2 or any(step.get("object_name") != new_target for step in observe_steps):
        raise ValueError(f"index={index}: oracle observe_target fields were not repaired")


def _domain_name(episode: dict[str, Any]) -> str:
    domains = _interactive(episode).get("interaction_domains")
    if domains == ["channel"]:
        return "channel"
    if domains == ["container"]:
        return "container"
    if domains == ["channel", "container"]:
        return "mixed"
    raise ValueError(f"Unknown V3 interaction domains: {domains!r}")


def _write_domain_file_or_copy(
    source_dir: Path, output_dir: Path, domain: str, episodes: list[dict[str, Any]]
) -> None:
    source_path = source_dir / f"{domain}.json"
    output_path = output_dir / f"{domain}.json"
    if domain != "channel" and source_path.exists():
        source_payload = _read_json(source_path)
        if _sha256_json(source_payload) != _sha256_json(episodes):
            raise ValueError(f"Source {domain}.json does not match input benchmark domain entries")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        return
    _write_json(output_path, episodes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-benchmark", type=Path, required=True)
    parser.add_argument("--repair-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.dry_run and args.output_root is None:
        raise ValueError("--output-root is required unless --dry-run is set")
    source_benchmark = args.input_benchmark.resolve()
    source_dir = source_benchmark.parent
    source_root = source_dir.parent
    repair_map_path = args.repair_map.resolve()
    input_sha256 = _sha256_path(source_benchmark)
    repair_map = _read_json(repair_map_path)
    expected_sha256 = repair_map.get("source_benchmark_sha256")
    if expected_sha256 != input_sha256:
        raise ValueError(
            "Repair map was audited against a different source benchmark: "
            f"expected={expected_sha256}, actual={input_sha256}"
        )
    repairs = repair_map.get("repairs")
    if not isinstance(repairs, list) or not repairs:
        raise ValueError("Repair map has no repairs list")
    by_index: dict[int, dict[str, Any]] = {}
    for repair in repairs:
        index = int(repair["benchmark_index"])
        if index in by_index:
            raise ValueError(f"Duplicate repair-map benchmark index {index}")
        if repair["old_selected_target"] == repair["new_selected_target"]:
            raise ValueError(f"index={index}: repair map contains a no-op target replacement")
        by_index[index] = repair

    benchmark = _read_json(source_benchmark)
    if not isinstance(benchmark, list):
        raise ValueError("Input benchmark must be a JSON list")
    if any(index < 0 or index >= len(benchmark) for index in by_index):
        raise ValueError("Repair map contains an out-of-range benchmark index")
    repaired = copy.deepcopy(benchmark)
    for index in sorted(by_index):
        repair = by_index[index]
        _validate_pre_repair_episode(repaired[index], repair)
        repaired[index] = _replace_exact_value(
            repaired[index], repair["old_selected_target"], repair["new_selected_target"]
        )
        _validate_post_repair_episode(repaired[index], repair)

    original_non_channel = [episode for episode in benchmark if _domain_name(episode) != "channel"]
    repaired_non_channel = [episode for episode in repaired if _domain_name(episode) != "channel"]
    if _sha256_json(original_non_channel) != _sha256_json(repaired_non_channel):
        raise RuntimeError("Container or Mixed entries changed during Channel-only repair")
    domains = {
        domain: [episode for episode in repaired if _domain_name(episode) == domain]
        for domain in ("channel", "container", "mixed")
    }
    counts = {domain: len(episodes) for domain, episodes in domains.items()}
    if counts != {"channel": 1000, "container": 1000, "mixed": 1000}:
        raise ValueError(f"Unexpected V3 domain counts: {counts}")

    report = {
        "schema_version": "interactive_nav_v3_channel_target_identity_repair_v2",
        "source_benchmark": str(source_benchmark),
        "source_benchmark_sha256": input_sha256,
        "repair_map": str(repair_map_path),
        "repair_map_sha256": _sha256_path(repair_map_path),
        "selection_rule": repair_map.get("selection_rule"),
        "terminal_compatibility_rule": repair_map.get("terminal_compatibility_rule"),
        "summary": {
            "repaired_channel_entry_count": len(by_index),
            "case_type_counts": repair_map.get("summary", {}).get("case_type_counts"),
            "domain_counts": counts,
            "container_mixed_canonical_sha256": _sha256_json(repaired_non_channel),
            "paths_or_interactions_rebuilt": False,
        },
        "repairs": repairs,
    }
    if args.dry_run:
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    output_root = args.output_root.resolve()
    if output_root == source_root:
        raise ValueError("Refusing to overwrite the source benchmark root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output root: {output_root}")
    balanced_dir = output_root / "balanced"
    _write_json(balanced_dir / "benchmark.json", repaired)
    for domain, episodes in domains.items():
        _write_domain_file_or_copy(source_dir, balanced_dir, domain, episodes)
    if (source_root / "config.resolved.yaml").exists():
        shutil.copy2(source_root / "config.resolved.yaml", output_root / "config.resolved.yaml")
    shutil.copy2(repair_map_path, output_root / "repair_target_map.json")
    report["output_benchmark"] = str((balanced_dir / "benchmark.json").resolve())
    report["output_benchmark_sha256"] = _sha256_path(balanced_dir / "benchmark.json")
    _write_json(output_root / "repair_manifest.json", report)
    print(json.dumps(report["summary"] | {"output_root": str(output_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
