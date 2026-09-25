#!/usr/bin/env python3
"""Aggregate a paired ablation pool under the evaluator's V4 paper metrics."""

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablations.paper_report import aggregate, comparison_ready


FIELDS = (
    "nav_success", "success", "required_interaction_success",
    "required_interaction_completion_fraction", "interaction_requirement",
    "reference_path_length_m", "navigation_path_length_m", "spl",
    "interaction_precision_episode", "episode_total_cost", "paper_metric_schema_version",
    "paper_metric_config", "interaction_action_count", "repeated_interaction_attempt_count",
    "error_interaction_attempt_count", "terminal_reason", "scoring_eligible",
)
VARIANTS = (
    ("full", "Full"),
    ("no_interaction_graph", "Flat Object Memory"),
    ("no_task_decision", "Greedy Selection"),
    ("no_outcome_update", "Perception-only Update"),
)


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def model_metrics(attempt):
    counts = {}
    errors = {}
    tokens = 0
    path = attempt / "mllm_metrics.jsonl"
    if not path.is_file():
        return {"roles": {}, "errors": {}, "tokens": 0}
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        role = str(row.get("role") or "unknown")
        counts[role] = counts.get(role, 0) + 1
        if row.get("error"):
            errors[role] = errors.get(role, 0) + 1
        tokens += int(row.get("total_tokens") or 0)
    return {"roles": dict(counts), "errors": dict(errors), "tokens": tokens}


def episode_row(variant, summary_path):
    batch = read_json(summary_path)
    raw_path = batch.get("episode_result_path")
    result_path = Path(raw_path) if raw_path else None
    result = read_json(result_path).get("result", {}) if result_path and result_path.is_file() else {}
    attempt_dir = batch.get("attempt_dir")
    row = {
        "variant": variant,
        "episode_index": batch.get("episode_index"),
        "scene_index": int(batch["episode_index"]) - 2000,
        "completed": batch.get("completed") is True,
        "runner_exit_code": batch.get("runner_exit_code"),
        "result_path": str(result_path) if result_path else None,
        "mllm": model_metrics(Path(attempt_dir)) if attempt_dir else {"roles": {}, "errors": {}, "tokens": 0},
    }
    row.update({field: result.get(field) for field in FIELDS})
    return row


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def summarize(pool):
    root = Path(pool).resolve()
    manifest = read_json(root / "pool_manifest.json")
    jobs = manifest.get("jobs") or []
    variants = list(dict.fromkeys(variant for variant, _ in jobs))
    rows = []
    for variant in variants:
        for summary in sorted((root / variant).glob("episode_*/batch_task_summary.json")):
            rows.append(episode_row(variant, summary))
    groups = {variant: aggregate([row for row in rows if row["variant"] == variant])
              for variant in variants}
    indices = manifest.get("config", {}).get("episode_indices", [])
    expected = len(indices)
    complete = comparison_ready(groups, variants, indices)
    report = {
        "complete": complete,
        "expected_scenes": expected,
        "expected_jobs": len(jobs),
        "reported_jobs": len(rows),
        "sample_episode_indices": manifest.get("config", {}).get("episode_indices", []),
        "sample_scene_indices": [int(i) - 2000 for i in manifest.get("config", {}).get("episode_indices", [])],
        "selection_rule": manifest.get("selection_rule"),
        "config": manifest.get("config", {}),
        "groups": groups,
        "episodes": rows,
    }
    (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    columns = ["variant", "scene_index", "episode_index", "completed", "runner_exit_code", *FIELDS, "result_path"]
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: json.dumps(row.get(column), ensure_ascii=False)
                          if column == "paper_metric_config" else row.get(column)
                          for column in columns} for row in rows)
    lines = ["# Paired V4 ablation results", "",
             f"Status: {'ready' if complete else 'incomplete or incompatible'}; "
             f"{len(rows)}/{len(jobs)} jobs reported.",
             "Metric schema: interactive_nav_v3_paper_metrics_v4.",
             "SR uses nav_success; SPL is recomputed from saved paths; ISR is the "
             "per-episode required-effect completion fraction; IP is an episode macro-average; "
             "Cost is normalized to [0,1].", "",
             "| Method | Eligible | SR ↑ | SPL ↑ | ISR ↑ | IP ↑ | Cost ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    labels = dict(VARIANTS)
    for variant in variants:
        group = groups[variant]
        def paper_value(name):
            return group.get(name) if group["paper_ready"] else None
        lines.append("| " + " | ".join([
            labels.get(variant, variant), f"{group['eligible']}/{group['reported']}",
            pct(paper_value("paper_sr")), num(paper_value("paper_spl"), 3),
            pct(paper_value("paper_isr")), pct(paper_value("paper_ip")),
            num(paper_value("paper_total_cost")),
        ]) + " |")
    lines += ["", "A complete comparison requires V4 results, all five metrics, the same "
              "eligible episodes and identical Cost settings in every variant.", "",
              "## Run audit", ""]
    for variant in variants:
        group = groups[variant]
        lines.append(f"- {labels.get(variant, variant)}: completed={group['completed']}, "
                     f"eligible={group['eligible']}, V4-ready={group['paper_ready']}, "
                     f"MLLM errors={sum(group['mllm_errors'].values())}, "
                     f"tokens={group['mllm_tokens']}, terminations={group['terminal_reasons']}.")
    (root / "comparison.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    args = parser.parse_args()
    report = summarize(args.pool)
    print(json.dumps({"complete": report["complete"], "reported": report["reported_jobs"],
                      "expected": report["expected_jobs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
