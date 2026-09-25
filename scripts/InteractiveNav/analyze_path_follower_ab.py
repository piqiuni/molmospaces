#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def episode_result(root: Path, episode_index: int) -> dict | None:
    task_dir = root / f"episode_{episode_index}"
    episode_dir = task_dir / "attempt_001" / "eval"
    result = episode_dir / "results.json"
    if not result.is_file():
        summary = task_dir / "batch_task_summary.json"
        if summary.is_file():
            record = json.loads(summary.read_text(encoding="utf-8"))
            return {"_infrastructure_error": record.get("error") or "missing episode result"}
        return None
    rows = json.loads(result.read_text(encoding="utf-8"))
    return rows[0] if len(rows) == 1 else None


def format_episode(result: dict | None) -> str:
    if result is None:
        return "运行中"
    if "_infrastructure_error" in result:
        return f"运行异常/{result['_infrastructure_error']}"
    steps = result.get("step_count")
    applied_steps = result.get("applied_action_step_count", "?")
    distance = result.get("navigation_path_length_m")
    success = int(bool(result.get("success")))
    reason = result.get("terminal_reason") or "?"
    distance_text = f"{distance:.2f}m" if distance is not None else "?"
    spl = result.get("spl")
    spl_text = f"{spl:.3f}" if spl is not None else "?"
    interactions = result.get("correct_interaction_action_count", "?")
    elapsed = result.get("elapsed_seconds")
    elapsed_text = f"{elapsed:.0f}s" if elapsed is not None else "?"
    return f"{success}/{steps}/{applied_steps}/{distance_text}/{spl_text}/{interactions}/{elapsed_text}/{reason}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired V3 path-follower/DWA episodes")
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--dwa", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    totals = {"new": {"success": 0, "stalled": 0, "reported": 0, "infra_failed": 0,
                      "spl_sum": 0.0, "distance_sum_m": 0.0, "elapsed_sum_s": 0.0,
                      "correct_interactions": 0, "applied_step_sum": 0},
              "dwa": {"success": 0, "stalled": 0, "reported": 0, "infra_failed": 0,
                      "spl_sum": 0.0, "distance_sum_m": 0.0, "elapsed_sum_s": 0.0,
                      "correct_interactions": 0, "applied_step_sum": 0}}
    print("episode | new (success/observation steps/applied steps/distance/SPL/correct interactions/elapsed/termination) "
          "| baseline (success/observation steps/applied steps/distance/SPL/correct interactions/elapsed/termination)")
    print("--- | --- | ---")
    for episode_index in args.episodes:
        results = {"new": episode_result(args.new, episode_index),
                   "dwa": episode_result(args.dwa, episode_index)}
        print(f"{episode_index} | {format_episode(results['new'])} | {format_episode(results['dwa'])}")
        for variant, result in results.items():
            if result is not None:
                totals[variant]["reported"] += 1
                if "_infrastructure_error" in result:
                    totals[variant]["infra_failed"] += 1
                    continue
                totals[variant]["success"] += int(bool(result.get("success")))
                totals[variant]["stalled"] += int(result.get("terminal_reason") == "policy_exploration_stalled")
                totals[variant]["spl_sum"] += float(result.get("spl") or 0.0)
                totals[variant]["distance_sum_m"] += float(result.get("navigation_path_length_m") or 0.0)
                totals[variant]["elapsed_sum_s"] += float(result.get("elapsed_seconds") or 0.0)
                totals[variant]["correct_interactions"] += int(result.get("correct_interaction_action_count") or 0)
                totals[variant]["applied_step_sum"] += int(result.get("applied_action_step_count") or 0)
    for variant in totals:
        count = totals[variant]["reported"] - totals[variant]["infra_failed"]
        if count:
            totals[variant]["mean_spl"] = round(totals[variant]["spl_sum"] / count, 3)
            totals[variant]["mean_distance_m"] = round(totals[variant]["distance_sum_m"] / count, 2)
            totals[variant]["mean_elapsed_s"] = round(totals[variant]["elapsed_sum_s"] / count, 1)
            totals[variant]["mean_applied_steps"] = round(totals[variant]["applied_step_sum"] / count, 1)
    print("totals", json.dumps(totals, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
