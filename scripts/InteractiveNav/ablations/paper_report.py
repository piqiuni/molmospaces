"""Aggregate ablation episodes with the evaluator's current paper formulas."""

from collections import Counter
import math

from evaluation.benchmark_metrics import _group_summary


def aggregate(rows):
    eligible = [row for row in rows if row.get("completed") is True
                and row.get("scoring_eligible") is True]
    paper = _group_summary(eligible) if eligible else {}
    result = {
        "reported": len(rows),
        "completed": sum(row.get("completed") is True for row in rows),
        "eligible": len(eligible),
        "reported_indices": sorted(int(row["episode_index"]) for row in rows),
        "eligible_indices": sorted(int(row["episode_index"]) for row in eligible),
        "terminal_reasons": dict(Counter(str(row.get("terminal_reason") or "unknown") for row in rows)),
        "mllm_roles": dict(sum((Counter(row["mllm"]["roles"]) for row in rows), Counter())),
        "mllm_errors": dict(sum((Counter(row["mllm"]["errors"]) for row in rows), Counter())),
        "mllm_tokens": sum(row["mllm"]["tokens"] for row in rows),
    }
    for name in ("interaction_action_count", "repeated_interaction_attempt_count",
                 "error_interaction_attempt_count"):
        result[name + "_sum"] = sum(int(row.get(name) or 0) for row in eligible)
    result.update(paper)
    result["paper_ready"] = bool(eligible) and all((
        paper.get("paper_metric_schema_current"),
        paper.get("paper_metric_config_consistent"),
        paper.get("paper_cost_config_valid"),
        *(isinstance(paper.get(name), (int, float)) and math.isfinite(paper[name])
          for name in ("paper_sr", "paper_spl", "paper_isr", "paper_ip", "paper_total_cost")),
    ))
    return result


def comparison_ready(groups, variants, expected_indices):
    expected_indices = sorted(int(index) for index in expected_indices)
    if not variants or not expected_indices or len(expected_indices) != len(set(expected_indices)):
        return False
    selected = [groups[variant] for variant in variants]
    if not all(group["reported"] == group["completed"] == len(expected_indices)
               and group["reported_indices"] == expected_indices
               and group["paper_ready"] for group in selected):
        return False
    indices = selected[0]["eligible_indices"]
    config = selected[0]["paper_metric_config"]
    return bool(indices) and all(
        group["eligible_indices"] == indices
        and group["paper_metric_config"] == config
        for group in selected
    )
