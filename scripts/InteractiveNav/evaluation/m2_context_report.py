#!/usr/bin/env python3
"""Report fixed-label paired M2 context ablations, without retuning labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.remove(str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}


def _percentile(values: list[float], quantile: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * quantile) - 1)] if values else None


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values), "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _strict(case: dict[str, Any]) -> bool:
    return bool((case.get("source", {}).get("reconstruction") or {}).get("strict_version_aligned"))


def _index_unique(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity or identity in result:
            raise ValueError(f"Missing/duplicate {key}: {identity}")
        result[identity] = row
    return result


def _score_case(
    case: dict[str, Any], prediction: dict[str, Any] | None, annotation: dict[str, Any],
) -> dict[str, Any]:
    candidate_ids = {candidate["id"] for candidate in case["request"]["candidates"]}
    acceptable = set(annotation.get("acceptable_top1_ids") or [])
    forbidden = set(annotation.get("forbidden_ids") or [])
    error = str((prediction or {}).get("transport_error") or "")
    response, schema_error = replay.validate_response((prediction or {}).get("response"), sorted(candidate_ids))
    valid = prediction is not None and not error and response is not None
    top1 = response["ranked_ids"][0] if valid else ""
    attempts = (prediction or {}).get("attempts") or []
    # The replay's prompt_tokens includes retries. Separate last-attempt input
    # size from total token consumption, rather than inflating context length.
    actual_prompt_tokens = (attempts[-1].get("prompt_tokens") if attempts else (prediction or {}).get("prompt_tokens")) or 0
    return {
        "case_id": case["case_id"], "base_case_id": case["base_case_id"], "arm": case["arm"],
        "split": annotation["split"], "strict_version_aligned": _strict(case),
        "main_score_eligible": bool(annotation.get("main_score_eligible")),
        "label_kind": annotation.get("label_kind"), "single_candidate": bool(annotation.get("single_candidate")),
        "prediction_present": prediction is not None, "valid": valid, "top1": top1,
        "accepted": valid and top1 in acceptable, "forbidden_top1": valid and top1 in forbidden,
        "forbidden_exposed": bool(candidate_ids & forbidden),
        "candidate_recall": bool(candidate_ids & acceptable),
        "acceptable_candidate_ids_available": sorted(candidate_ids & acceptable),
        "acceptable_candidate_ids_missing": sorted(acceptable - candidate_ids),
        "candidate_count": len(candidate_ids), "transport_error": error,
        "schema_error": schema_error if prediction is not None and not error else "",
        "input_tokens_actual": float(actual_prompt_tokens) if actual_prompt_tokens > 0 else None,
        "input_tokens_estimate": case.get("input_tokens_estimate"),
        "prompt_tokens_all_attempts": (prediction or {}).get("prompt_tokens", 0),
        "total_tokens_all_attempts": (prediction or {}).get("total_tokens", 0),
        "retry_count": (prediction or {}).get("retry_count", 0),
        "latency_s": prediction.get("latency_s") if prediction is not None else None,
        "total_latency_s": prediction.get("total_latency_s", prediction.get("latency_s")) if prediction is not None else None,
    }


def _subset(rows: list[dict[str, Any]], split: str, alignment: str) -> list[dict[str, Any]]:
    return [row for row in rows if (split == "all" or row["split"] == split) and (
        alignment == "all" or row["strict_version_aligned"] == (alignment == "strict_version_aligned")
    )]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    main = [row for row in rows if row["main_score_eligible"]]
    exposed = [row for row in rows if row["forbidden_exposed"]]
    stage = [row for row in main if row["label_kind"] == "public_stage_contract"]
    single = [row for row in rows if row["single_candidate"]]
    distributions = {
        key: _distribution([float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))])
        for key in ("input_tokens_actual", "input_tokens_estimate", "latency_s", "total_latency_s", "candidate_count")
    }
    return {
        "cases": len(rows), "predictions_received": sum(row["prediction_present"] for row in rows),
        "missing_predictions": sum(not row["prediction_present"] for row in rows),
        "main_acceptable_top1": _ratio(sum(row["accepted"] for row in main), len(main)),
        "candidate_recall": _ratio(sum(row["candidate_recall"] for row in main), len(main)),
        "upstream_candidate_recall_losses": sum(not row["candidate_recall"] for row in main),
        "selection_or_request_losses_with_acceptable_candidate": sum(row["candidate_recall"] and not row["accepted"] for row in main),
        "forbidden_top1_exposed": _ratio(sum(row["forbidden_top1"] for row in exposed), len(exposed)),
        "schema_valid": _ratio(sum(row["valid"] for row in rows), len(rows)),
        "schema_error_count": sum(bool(row["schema_error"]) for row in rows),
        "transport_error_count": sum(bool(row["transport_error"]) for row in rows),
        "stage_acceptance": _ratio(sum(row["accepted"] for row in stage), len(stage)),
        "single_candidate_validity": _ratio(sum(row["valid"] for row in single), len(single)),
        "retry_count": sum(row["retry_count"] for row in rows),
        "prompt_tokens_all_attempts": sum(row["prompt_tokens_all_attempts"] for row in rows),
        "total_tokens_all_attempts": sum(row["total_tokens_all_attempts"] for row in rows),
        "distributions": distributions,
        "error_counts": dict(Counter(row["transport_error"] or row["schema_error"] for row in rows if row["transport_error"] or row["schema_error"])),
    }


def paired(full: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["base_case_id"]: row for row in full if row["main_score_eligible"]}
    right = {row["base_case_id"]: row for row in other if row["main_score_eligible"]}
    if set(left) != set(right):
        raise ValueError("Paired arms do not have the same fixed eligible case set")
    wins = sorted(case_id for case_id in left if left[case_id]["accepted"] and not right[case_id]["accepted"])
    losses = sorted(case_id for case_id in left if right[case_id]["accepted"] and not left[case_id]["accepted"])
    return {
        "denominator": len(left), "full_only_accepted": len(wins), "ablation_only_accepted": len(losses),
        "both_accepted": sum(left[key]["accepted"] and right[key]["accepted"] for key in left),
        "neither_accepted": sum(not left[key]["accepted"] and not right[key]["accepted"] for key in left),
        "net_full_gain": len(wins) - len(losses),
        "net_full_gain_rate": (len(wins) - len(losses)) / len(left) if left else None,
        "full_win_case_ids": wins, "full_loss_case_ids": losses,
    }


def guard_shadow(
    rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]], diagnostics: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    replay._install_runtime_import_paths()
    from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
    from semantic_decision_py_pkg.model_policy import ModelPolicyClient, ModelPolicyConfig

    client = ModelPolicyClient(ModelPolicyConfig(mode="disabled"))
    shadows = []
    for row in rows:
        if not row["valid"]:
            continue
        case = cases[row["case_id"]]
        pool = "legacy" if case["arm"] == "legacy_candidate_pool" else "all_actions_12_frontiers"
        diagnostic = (diagnostics.get(case["base_case_id"], {}).get("pools") or {}).get(pool)
        if diagnostic is None:
            continue
        groups = {candidate["id"]: [BehaviorCandidate(
            candidate_id=candidate["id"], behavior_type="EXPLORE" if candidate.get("subject_type") == "frontier" else "NAVIGATE",
            source="guard_shadow", target_id=str(candidate.get("subject_id") or ""), target_name="",
            metadata={"room_status": candidate.get("room_status")},
        )] for candidate in case["request"]["candidates"]}
        client.last_pre_score_guard = ""
        chosen = client._apply_pre_score_guard(row["top1"], groups, {
            "candidate_pre_scores": diagnostic.get("quality_by_id") or {},
            "candidate_decision_hints": diagnostic.get("decision_hint_by_id") or {},
        })
        annotation = annotations[case["base_case_id"]]
        shadows.append({
            "arm": case["arm"], "base_case_id": case["base_case_id"], "raw_top1": row["top1"],
            "shadow_top1": chosen, "changed": chosen != row["top1"], "guard_reason": client.last_pre_score_guard,
            "main_score_eligible": row["main_score_eligible"], "raw_accepted": row["accepted"],
            "shadow_accepted": chosen in set(annotation.get("acceptable_top1_ids") or []),
            "shadow_forbidden": chosen in set(annotation.get("forbidden_ids") or []),
        })
    return {
        "note": "Separate offline shadow only: production guard, default margin 0.75, public room_status and frozen diagnostics. No simulator rerun or dispatch validation; never substituted into raw main scores.",
        "by_arm": {arm: {
            "evaluated_valid_predictions": sum(row["arm"] == arm for row in shadows),
            "changed": sum(row["arm"] == arm and row["changed"] for row in shadows),
            "main_recovered": sum(row["arm"] == arm and row["main_score_eligible"] and row["shadow_accepted"] and not row["raw_accepted"] for row in shadows),
            "main_harmed": sum(row["arm"] == arm and row["main_score_eligible"] and row["raw_accepted"] and not row["shadow_accepted"] for row in shadows),
        } for arm in sorted({row["arm"] for row in shadows})},
        "changed_cases": [row for row in shadows if row["changed"]],
    }


def _fraction(metric: dict[str, Any]) -> str:
    return f"{metric['numerator']}/{metric['denominator']} ({metric['rate']:.1%})" if metric["denominator"] else "—"


def generate(inputs: Path, run: Path, annotations_path: Path, output_dir: Path, *, include_guard_shadow: bool = False) -> dict[str, Any]:
    cases_path = inputs / "cases.jsonl"
    predictions_path = run / "predictions.jsonl"
    input_manifest = json.loads((inputs / "manifest.json").read_text())
    run_manifest = json.loads((run / "manifest.json").read_text())
    cases = _index_unique(_rows(cases_path), "case_id")
    predictions = _index_unique(_rows(predictions_path), "case_id")
    annotations = _index_unique(_rows(annotations_path), "case_id")
    unexpected = set(predictions) - set(cases)
    if unexpected:
        raise ValueError(f"Predictions not present in frozen inputs: {sorted(unexpected)[:3]}")
    rows = [_score_case(case, predictions.get(case_id), annotations[case["base_case_id"]]) for case_id, case in cases.items()]
    arms = sorted({row["arm"] for row in rows})
    arm_rows = {arm: [row for row in rows if row["arm"] == arm] for arm in arms}
    alignments = ("all", "strict_version_aligned", "lagged_or_incomplete")
    splits = ("all", "dev", "holdout")
    summaries = {arm: {split: {alignment: summarize(_subset(arm_rows[arm], split, alignment)) for alignment in alignments} for split in splits} for arm in arms}
    comparisons = {arm: {split: {alignment: paired(
        _subset(arm_rows.get("full_context", []), split, alignment), _subset(arm_rows[arm], split, alignment),
    ) for alignment in alignments} for split in splits} for arm in arms if arm != "full_context"}
    result = {
        "schema_version": "m2_context_comparison_v1", "metric": "discriminative_public_rubric_top1_acceptance",
        "scope": "raw Qwen output, fixed prediction-blind public-policy-proxy labels; not navigation SR",
        "input_manifest": input_manifest, "run_manifest": run_manifest,
        "file_sha256": {"inputs": _sha(cases_path), "predictions": _sha(predictions_path), "annotations": _sha(annotations_path), "report_generator": _sha(Path(__file__))},
        "by_arm": summaries, "paired_full_vs_ablation": comparisons,
        "case_scores": rows,
        "notes": [
            "compact_control is a compact projection of the same expanded pool, not the original historical HTTP request.",
            "All arms use the same union labels; only legacy_candidate_pool changes the pool. It is a compound pool-strategy ablation, not simply 8 versus 12 candidates.",
            "no_recent_decisions removes the last 30 explicit events, not aggregate candidate history; no_room_route removes route order/timing, not all visited-room state.",
            "Missing predictions, request failures, schema errors and missing all acceptable candidates remain failures in the fixed main denominator.",
            "Exact-version-aligned reconstructed inputs are not exact recorded HTTP requests. Report lagged/incomplete separately.",
            "Repeated decisions within an episode are dependent; no independent-sample significance claim is made.",
        ],
    }
    if include_guard_shadow:
        diagnostics = _index_unique(_rows(inputs / "curation_diagnostics.jsonl"), "case_id")
        result["guard_shadow"] = guard_shadow(rows, cases, diagnostics, annotations)
    lines = ["# Qwen M2 上下文消融", "", "以下为固定公开事实 rubric 的合理 top-1 接受率，**不是导航成功率（SR）**。失败请求、无效回答和候选池漏掉合理动作均计入固定分母。", "", "| Arm | 全量主指标 | Dev | Holdout | 严格版本对齐 | 滞后/不完整 | 候选召回 |", "|---|---|---|---|---|---|---|"]
    for arm in arms:
        stats = summaries[arm]
        lines.append(f"| {arm} | {_fraction(stats['all']['all']['main_acceptable_top1'])} | {_fraction(stats['dev']['all']['main_acceptable_top1'])} | {_fraction(stats['holdout']['all']['main_acceptable_top1'])} | {_fraction(stats['all']['strict_version_aligned']['main_acceptable_top1'])} | {_fraction(stats['all']['lagged_or_incomplete']['main_acceptable_top1'])} | {_fraction(stats['all']['all']['candidate_recall'])} |")
    lines.extend(["", "| Arm | 禁止动作（暴露样本） | schema错误 | 请求错误 | 缺少预测 | 输入token p50/p95 | 耗时秒 p50/p95 |", "|---|---|---|---|---|---|---|"])
    for arm in arms:
        stat = summaries[arm]["all"]["all"]
        tokens = stat["distributions"]["input_tokens_actual"]
        latency = stat["distributions"]["total_latency_s"]
        lines.append(f"| {arm} | {_fraction(stat['forbidden_top1_exposed'])} | {stat['schema_error_count']} | {stat['transport_error_count']} | {stat['missing_predictions']} | {tokens['p50']} / {tokens['p95']} | {latency['p50']} / {latency['p95']} |")
    lines.extend(["", "## 配对变化：full_context 相对于各消融", "", "| 对照 arm | 全量仅full合理 | 仅消融合理 | 净增 | Holdout净增 |", "|---|---|---|---|---|"])
    for arm, by_split in comparisons.items():
        pair = by_split["all"]["all"]
        lines.append(f"| {arm} | {pair['full_only_accepted']} | {pair['ablation_only_accepted']} | {pair['net_full_gain']} / {pair['denominator']} | {by_split['holdout']['all']['net_full_gain']} / {by_split['holdout']['all']['denominator']} |")
    lines.extend(["", "## 解释边界", "", *[f"- {note}" for note in result["notes"]], "", "输入token统计使用最后一次请求的usage，不把重试累计token误当上下文长度；耗时包括重试。分split/版本层、阶段合同、单候选有效率、错误类型及具体配对case见comparison.json。"])
    if include_guard_shadow:
        lines.extend(["", "## Guard shadow（不并入原始主指标）", "", result["guard_shadow"]["note"], "", "| Arm | 改写 | 主指标挽回 | 主指标损害 |", "|---|---|---|---|"])
        for arm, stat in result["guard_shadow"]["by_arm"].items():
            lines.append(f"| {arm} | {stat['changed']} | {stat['main_recovered']} | {stat['main_harmed']} |")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("comparison.json", "report.md")):
        raise FileExistsError("Report already exists; use a fresh output directory")
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--guard-shadow", action="store_true")
    args = parser.parse_args()
    result = generate(args.inputs, args.run, args.annotations, args.output_dir, include_guard_shadow=args.guard_shadow)
    print(json.dumps({"arms": len(result["by_arm"]), "scored_input_rows": len(result["case_scores"]), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
