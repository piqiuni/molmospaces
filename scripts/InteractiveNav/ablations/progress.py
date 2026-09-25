"""Compact, read-only progress view for running ablation pools."""

import json
import math
from pathlib import Path
from statistics import mean


LABELS = {
    "full": "Full",
    "no_interaction_graph": "Flat",
    "no_task_decision": "Greedy",
    "no_outcome_update": "Perception-only",
}
PAPER_SCHEMA = "interactive_nav_v3_paper_metrics_v4"
METRICS = (
    ("TaskSR", "task_success", ".3f"),
    ("ICS", "interaction_conditioned_success", ".3f"),
    ("NavSR", "nav_success", ".3f"),
    ("ISR", "required_interaction_completion_fraction", ".3f"),
    ("SPL", "spl", ".3f"),
    ("IP", "interaction_precision_episode", ".3f"),
    ("Cost", "episode_total_cost", ".3f"),
    ("Steps", "step_count", ".1f"),
    ("Eval", "elapsed_seconds", ".1f"),
)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def episode_results(pool, variant):
    """Read only completed episode summaries; tolerate files being written now."""
    completed = 0
    eligible = []
    for summary_path in sorted((Path(pool) / variant).glob("episode_*/batch_task_summary.json")):
        summary = read_json(summary_path)
        if summary.get("completed") is not True:
            continue
        completed += 1
        result_path = summary.get("episode_result_path")
        result = read_json(result_path).get("result", {}) if result_path else {}
        if not isinstance(result, dict) or result.get("scoring_eligible") is not True:
            continue
        if result.get("paper_metric_schema_version") != PAPER_SCHEMA:
            continue
        if all(isinstance(result.get(field), (int, float))
               and math.isfinite(result[field]) for _, field, _ in METRICS):
            eligible.append(result)
    return completed, eligible


def format_variant(pool, variant, state, planned):
    variant_state = state.get("per_variant", {}).get(variant, {})
    reported = int(variant_state.get("reported", 0))
    active = sum(job[0] == variant for job in state.get("active", {}).values())
    queued = max(0, planned - reported - active)
    completed, eligible = episode_results(pool, variant)
    failed = max(0, reported - completed)
    elapsed = float(state.get("elapsed_sec") or 0) / 60
    title = (f"[消融 {LABELS.get(variant, variant)} {elapsed:.1f}min] "
             f"完成 {completed}/{planned} | 运行/收尾 {active} | 排队 {queued} | 运行异常 {failed}")
    if not eligible:
        return title + "\n  已完成均值：暂无可评分的 v4 结果"
    metrics = " | ".join(
        f"{label}={format(mean(row[field] for row in eligible), spec)}{'s' if label == 'Eval' else ''}"
        for label, field, spec in METRICS
    )
    return title + f"\n  已完成均值（n={len(eligible)}）：{metrics}"


def format_pool(pool):
    pool = Path(pool)
    manifest = read_json(pool / "pool_manifest.json")
    state = read_json(pool / "pool_status.json")
    jobs = manifest.get("jobs") or []
    variants = list(dict.fromkeys(job[0] for job in jobs))
    if not variants:
        return f"{pool}: 等待任务清单"
    if not state:
        return f"{pool}: 等待启动状态"
    return "\n".join(format_variant(pool, variant, state,
                                    sum(job[0] == variant for job in jobs))
                     for variant in variants)
