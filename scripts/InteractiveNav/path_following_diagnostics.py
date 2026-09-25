#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median


def path_distance(position: tuple[float, float], points: list[tuple[float, float]]) -> float:
    nearest = math.inf
    for first, second in zip(points, points[1:]):
        delta_x = second[0] - first[0]
        delta_y = second[1] - first[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared < 1e-12:
            continue
        fraction = max(0.0, min(1.0, (
            (position[0] - first[0]) * delta_x +
            (position[1] - first[1]) * delta_y
        ) / length_squared))
        nearest = min(nearest, math.dist(position, (
            first[0] + fraction * delta_x,
            first[1] + fraction * delta_y,
        )))
    return nearest


def summarize(episode_dir: Path) -> dict:
    attempt_dir = episode_dir / "attempt_001"
    debug_dir = attempt_dir / "debug"
    plan_file = debug_dir / "move_base_plans.csv"
    status_file = debug_dir / "move_base_status.csv"
    trajectory_file = debug_dir / "trajectory.csv"
    if not all(path.is_file() for path in (plan_file, status_file, trajectory_file)):
        return {"error": "recorded debug CSVs unavailable"}
    applied_step_limit = None
    result_file = attempt_dir / "eval" / "results.json"
    if result_file.is_file():
        try:
            results = json.loads(result_file.read_text(encoding="utf-8"))
            if len(results) == 1 and results[0].get("status") == "complete":
                applied_step_limit = int(results[0]["applied_action_step_count"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    paths: dict[int, dict] = {}
    with plan_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["plan_type"] != "global" or row["frame_id"] != "tf_frame_map":
                continue
            message_index = int(row["message_index"])
            record = paths.setdefault(message_index, {"step": int(row["step_id"]), "points": []})
            record["points"].append((float(row["x"]), float(row["y"])))
    plans = [dict(record, message_index=index) for index, record in paths.items()
             if len(record["points"]) >= 2]
    raw_file = debug_dir / "raw" / "step_boundaries.jsonl.gz"
    if raw_file.is_file():
        empty_messages = set()
        with gzip.open(raw_file, "rt", encoding="utf-8") as stream:
            for line in stream:
                published = (json.loads(line).get("global_plan") or {})
                if (published.get("frame_id") != "tf_frame_map" or
                        published.get("poses") != []):
                    continue
                message_index = published.get("message_index")
                if message_index is None or message_index in empty_messages:
                    continue
                empty_messages.add(message_index)
                plans.append({"step": int(published["step_id"]),
                              "message_index": int(message_index), "points": []})
    plans.sort(key=lambda record: (record["step"], record["message_index"]))
    with status_file.open(newline="", encoding="utf-8") as stream:
        statuses = sorted(((int(row["step_id"]), row["status_name"])
                           for row in csv.DictReader(stream)), key=lambda item: item[0])
    distances = []
    plan_index = 0
    status_index = 0
    active = False
    latest_path: list[tuple[float, float]] = []
    with trajectory_file.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step_id"])
            if applied_step_limit is not None and step > applied_step_limit:
                continue
            while plan_index < len(plans) and plans[plan_index]["step"] <= step:
                latest_path = plans[plan_index]["points"]
                plan_index += 1
            while status_index < len(statuses) and statuses[status_index][0] <= step:
                active = statuses[status_index][1] == "ACTIVE"
                status_index += 1
            if active and latest_path:
                deviation = path_distance((float(row["x"]), float(row["y"])), latest_path)
                if math.isfinite(deviation):
                    distances.append(deviation)
    if not distances:
        return {"samples": 0, "reason": "no samples with active goal and global path"}
    distances.sort()
    return {
        "samples": len(distances),
        "applied_step_limit": applied_step_limit,
        "median_deviation_m": round(median(distances), 3),
        "p95_deviation_m": round(distances[int(0.95 * (len(distances) - 1))], 3),
        "fraction_over_0_2m": round(sum(value > 0.2 for value in distances) / len(distances), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorded map-frame global-path proximity diagnostic")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for index in args.episodes:
        print(index, json.dumps(summarize(args.run_dir / f"episode_{index}"), sort_keys=True))


if __name__ == "__main__":
    main()
