#!/usr/bin/env python3
"""Aggregate a four-variant mixed-100 pool without changing evaluator scores."""

from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import statistics


FIELDS = (
    "nav_success", "success", "task_success", "required_interaction_success", "spl",
    "interaction_precision_episode", "episode_total_cost", "navigation_path_length_m",
    "reference_path_length_m", "interaction_action_count", "valid_interaction_attempt_count",
    "error_interaction_attempt_count", "repeated_interaction_attempt_count",
    "task_irrelevant_interaction_attempt_count", "failed_interaction_attempt_count",
    "step_count", "episode_step_budget", "elapsed_seconds", "terminal_reason",
    "scoring_eligible",
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
    counts = Counter()
    errors = Counter()
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
        counts[role] += 1
        if row.get("error"):
            errors[role] += 1
        tokens += int(row.get("total_tokens") or 0)
    return {"roles": dict(counts), "errors": dict(errors), "tokens": tokens}


def episode_row(variant, summary_path):
    batch = read_json(summary_path)
    result_path = Path(batch.get("episode_result_path") or "")
    result = read_json(result_path).get("result", {}) if result_path.is_file() else {}
    row = {
        "variant": variant,
        "episode_index": batch.get("episode_index"),
        "scene_index": int(batch["episode_index"]) - 2000,
        "completed": bool(batch.get("completed")),
        "runner_exit_code": batch.get("runner_exit_code"),
        "result_path": str(result_path),
        "mllm": model_metrics(result_path.parents[3]) if result_path.is_file() else {"roles": {}, "errors": {}, "tokens": 0},
    }
    row.update({field: result.get(field) for field in FIELDS})
    return row


def aggregate(rows):
    eligible = [row for row in rows if row["completed"] and row.get("scoring_eligible") is True]
    result = {"reported": len(rows), "completed": sum(bool(row["completed"]) for row in rows),
              "eligible": len(eligible)}
    for field in FIELDS:
        values = [row[field] for row in eligible if isinstance(row.get(field), (int, float))]
        if values:
            result[field] = statistics.mean(values)
            result[field + "_sum"] = sum(values)
    result["termination_counts"] = dict(Counter(str(row.get("terminal_reason") or "unknown") for row in rows))
    result["mllm_roles"] = dict(sum((Counter(row["mllm"]["roles"]) for row in rows), Counter()))
    result["mllm_errors"] = dict(sum((Counter(row["mllm"]["errors"]) for row in rows), Counter()))
    result["mllm_tokens"] = sum(row["mllm"]["tokens"] for row in rows)
    return result


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    args = parser.parse_args()
    root = args.pool.resolve()
    manifest = read_json(root / "pool_manifest.json")
    rows = []
    for variant, _label in VARIANTS:
        for summary in sorted((root / variant).glob("episode_*/batch_task_summary.json")):
            rows.append(episode_row(variant, summary))
    groups = {variant: aggregate([row for row in rows if row["variant"] == variant])
              for variant, _label in VARIANTS}
    expected = len(manifest.get("selected_full_rows") or manifest.get("config", {}).get("episode_indices", []))
    complete = all(groups[variant]["reported"] == expected and groups[variant]["eligible"] == expected
                   for variant, _label in VARIANTS)
    report = {
        "complete": complete,
        "expected_scenes": expected,
        "expected_jobs": expected * len(VARIANTS),
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
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    lines = ["# Mixed-100（每10个取1个）四类消融结果", "",
             f"状态：{'400/400 场完成且有效。' if complete else '运行未完成，以下为当前已报告结果。'}",
             "抽样 episode：" + "、".join(map(str, report["sample_episode_indices"])) + "。",
             "对应 mixed 场景：" + "、".join(map(str, report["sample_scene_indices"])) + "。",
             "四类均在本轮运行；25 worker、动态 max2000、关闭录制。Full 为本轮实际运行结果。", "",
             "| 方法 | n | SR | ICS | ISR | SPL | IP宏平均 | 平均Cost | 交互数 | 重复数 | 失败数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = dict(VARIANTS)
    for variant, _label in VARIANTS:
        group = groups[variant]
        lines.append("| " + " | ".join([
            labels[variant], str(group["eligible"]), pct(group.get("nav_success")),
            pct(group.get("success")), pct(group.get("required_interaction_success")),
            num(group.get("spl"), 3), pct(group.get("interaction_precision_episode")),
            num(group.get("episode_total_cost")), num(group.get("interaction_action_count_sum"), 0),
            num(group.get("repeated_interaction_attempt_count_sum"), 0),
            num(group.get("failed_interaction_attempt_count_sum"), 0),
        ]) + " |")
    lines += ["", "SR=nav_success；ICS=result.success；ISR=required_interaction_success。",
              "IP 为逐场宏平均，Cost 和成功判定沿用 evaluator，未重新评分。", "",
              "## 终止原因", ""]
    for variant, _label in VARIANTS:
        lines.append(f"- {labels[variant]}：" + ", ".join(
            f"{key}={value}" for key, value in sorted(groups[variant]["termination_counts"].items())))
    lines += ["", "## 运行审计", "", f"计划 job：{expected * len(VARIANTS)}；已报告：{len(rows)}。",
              "抽样规则为 mixed block episode 2000–2999 中每10个取一个（2000 + 10k）。",
              "结果按四类共同场景逐场保存于 comparison.csv；模型调用与错误汇总于 comparison.json。", ""]
    (root / "comparison.md").write_text("\n".join(lines))
    print(json.dumps({"complete": complete, "reported": len(rows), "expected": expected * len(VARIANTS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
