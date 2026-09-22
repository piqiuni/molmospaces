#!/usr/bin/env python3
"""Compare complete model runs on one frozen M2 full-context dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.remove(str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.InteractiveNav.evaluation import m2_context_report as report


OTHER_CONTEXT_ARMS = {
    "compact_control", "no_recent_decisions", "no_room_route", "no_geometry",
    "no_room_frontiers", "no_key_objects", "with_pre_scores", "legacy_candidate_pool",
}
SPLITS = ("all", "dev", "holdout")
ALIGNMENTS = ("all", "strict_version_aligned", "lagged_or_incomplete")


def _base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_case_id") or str(row.get("case_id") or "").split("::")[-1])


def _frozen_cases(inputs: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], Path]:
    cases = report._index_unique(report._rows(inputs / "cases.jsonl"), "case_id")
    if not cases:
        raise ValueError("Frozen input dataset is empty")
    for case in cases.values():
        if case.get("arm") != "full_context" or not str(case["case_id"]).startswith("full_context::"):
            raise ValueError("The shared input must contain only full_context cases")
        if _base_id(case) != str(case["case_id"]).split("::", 1)[1]:
            raise ValueError("Inconsistent base_case_id in frozen input")
    label_path = inputs / "base_annotations.jsonl"
    if not label_path.exists():
        label_path = inputs / "annotations.jsonl"
    labels = {}
    for row in report._rows(label_path):
        base = _base_id(row)
        if not base or base in labels:
            raise ValueError(f"Missing/duplicate annotation base case: {base}")
        labels[base] = row
    bases = {_base_id(case) for case in cases.values()}
    if set(labels) != bases:
        raise ValueError("Frozen label and input case sets differ")
    # Candidate pools are fixed across models, so all acceptable label IDs
    # must refer to actual offered choices. No labels are rewritten here.
    for case in cases.values():
        label = labels[_base_id(case)]
        candidates = {candidate["id"] for candidate in case["request"]["candidates"]}
        if not set(label.get("acceptable_top1_ids") or []) <= candidates:
            raise ValueError(f"Label IDs outside shared pool: {case['case_id']}")
    return cases, labels, label_path


def _complete_predictions(
    path: Path, cases: dict[str, dict[str, Any]], *, allow_qwen_ablations: bool,
) -> tuple[dict[str, dict[str, Any]], int]:
    predictions = report._index_unique(report._rows(path), "case_id")
    selected, ignored = {}, 0
    bases = {_base_id(case) for case in cases.values()}
    for case_id, row in predictions.items():
        if case_id in cases:
            if row.get("arm") not in (None, "full_context"):
                raise ValueError(f"Prediction arm conflicts with case namespace: {case_id}")
            if row.get("base_case_id") and row["base_case_id"] != _base_id(cases[case_id]):
                raise ValueError(f"Prediction base_case_id mismatch: {case_id}")
            selected[case_id] = row
            continue
        namespace = case_id.split("::", 1)[0]
        if (
            allow_qwen_ablations and namespace in OTHER_CONTEXT_ARMS
            and _base_id(row) in bases and row.get("arm") in (None, namespace)
        ):
            ignored += 1
            continue
        raise ValueError(f"Unexpected prediction outside shared full_context: {case_id}")
    missing = set(cases) - set(selected)
    if missing:
        raise ValueError(f"Incomplete run: missing {len(missing)} of {len(cases)} cases; wait for completion")
    return selected, ignored


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = report.summarize(rows)
    for key in ("rate_limit_wait_s", "completion_tokens_actual"):
        result["distributions"][key] = report._distribution([
            float(row[key]) for row in rows if row.get(key) is not None
        ])
    return result


def _pair(reference: list[dict[str, Any]], model: list[dict[str, Any]]) -> dict[str, Any]:
    raw = report.paired(reference, model)
    return {
        "denominator": raw["denominator"],
        "reference_only_accepted": raw["full_only_accepted"],
        "model_only_accepted": raw["ablation_only_accepted"],
        "both_accepted": raw["both_accepted"], "neither_accepted": raw["neither_accepted"],
        "net_model_gain": -raw["net_full_gain"],
        "net_model_gain_rate": -raw["net_full_gain_rate"] if raw["net_full_gain_rate"] is not None else None,
        "model_win_case_ids": raw["full_loss_case_ids"],
        "model_loss_case_ids": raw["full_win_case_ids"],
    }


def _value(value: Any) -> str:
    return "—" if value is None else f"{value:.3f}"


def generate(inputs: Path, runs: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("Provide Qwen and at least one comparison model")
    references = [label for label in runs if "qwen" in label.casefold()]
    if len(references) != 1:
        raise ValueError("Exactly one run label must contain 'qwen' to identify the paired reference")
    reference = references[0]
    cases, labels, label_path = _frozen_cases(inputs)
    input_manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    scored, checks = {}, {}
    for model, path in runs.items():
        predictions, ignored = _complete_predictions(path, cases, allow_qwen_ablations=model == reference)
        rows = []
        for case_id, case in cases.items():
            prediction = predictions[case_id]
            row = report._score_case(case, prediction, labels[_base_id(case)])
            row["model_label"] = model
            row["rate_limit_wait_s"] = float(prediction.get("rate_limit_wait_s", 0.0) or 0.0)
            attempts = prediction.get("attempts") or []
            row["completion_tokens_actual"] = (
                attempts[-1].get("completion_tokens") if attempts else prediction.get("completion_tokens")
            )
            rows.append(row)
        scored[model] = rows
        checks[model] = {
            "predictions_path": str(path.resolve()), "predictions_sha256": report._sha(path),
            "matched_cases": len(predictions), "ignored_other_qwen_arms": ignored,
            "complete": True,
        }
    summaries = {model: {split: {alignment: _summary(report._subset(rows, split, alignment)) for alignment in ALIGNMENTS} for split in SPLITS} for model, rows in scored.items()}
    comparisons = {model: {split: {alignment: _pair(
        report._subset(scored[reference], split, alignment), report._subset(scored[model], split, alignment),
    ) for alignment in ALIGNMENTS} for split in SPLITS} for model in runs if model != reference}
    result = {
        "schema_version": "m2_context_model_comparison_v1",
        "metric": "discriminative_public_rubric_top1_acceptance",
        "scope": "Same frozen full_context, prompt and labels; raw model outputs only, no production guard substitution",
        "reference_model_label": reference, "case_count_per_model": len(cases),
        "fixed_primary_by_split": dict(Counter(label["split"] for label in labels.values() if label.get("main_score_eligible"))),
        "input_manifest": input_manifest,
        "file_sha256": {
            "cases": report._sha(inputs / "cases.jsonl"), "labels": report._sha(label_path),
            "input_manifest": report._sha(inputs / "manifest.json"),
            "report_generator": report._sha(Path(__file__)), "shared_scoring_module": report._sha(Path(report.__file__)),
        },
        "run_checks": checks, "by_model": summaries, "paired_vs_qwen": comparisons,
        "case_scores": scored,
        "notes": [
            "Same frozen acceptable IDs and main_score_eligible denominator for every model; request/schema failures remain failures on eligible cases.",
            "Policy-proxy acceptance is not navigation SR, human ground truth, or a closed-loop causal performance measurement.",
            "Historical old-context scores used different labels/candidate pools/denominators and are not directly comparable to this table.",
            "latency_s sums measured request latency over attempts, including transport/provider queue/generation; it is not pure GPU time.",
            "total_latency_s additionally includes client rate-limit waiting and retry overhead after the worker starts; waiting for a free worker before execution is not included.",
            "Different model providers and concurrency/rate limits confound latency comparisons; this is not an equal-hardware speed benchmark.",
            "Exact-version-aligned is not an exact historical HTTP request. Stage labels can conflict with stale phase hints; all frozen labels remain unchanged.",
        ],
    }
    lines = [
        "# M2 同新上下文跨模型对比", "",
        "相同冻结full_context、提示词和公开规则标签；主指标为合理top-1接受率，**不是导航SR**。仅比较raw模型输出，不混入guard shadow。", "",
        "| 模型 | 全量主指标 | Dev | Holdout | 严格对齐主指标 | 阶段合同 | schema有效 |", "|---|---|---|---|---|---|---|",
    ]
    for model in runs:
        stats = summaries[model]
        lines.append(f"| {model} | {report._fraction(stats['all']['all']['main_acceptable_top1'])} | {report._fraction(stats['dev']['all']['main_acceptable_top1'])} | {report._fraction(stats['holdout']['all']['main_acceptable_top1'])} | {report._fraction(stats['all']['strict_version_aligned']['main_acceptable_top1'])} | {report._fraction(stats['all']['all']['stage_acceptance'])} | {report._fraction(stats['all']['all']['schema_valid'])} |")
    lines.extend(["", "## 请求耗时与token", "", "| 模型 | service均值/p50/p95秒 | total均值/p50/p95秒 | 限流等待均值秒 | 平均输入/输出token | transport/schema错误 | 重试 |", "|---|---|---|---|---|---|---|"])
    for model in runs:
        stats = summaries[model]["all"]["all"]
        distributions = stats["distributions"]
        service = "/".join(_value(distributions["latency_s"][key]) for key in ("mean", "p50", "p95"))
        total = "/".join(_value(distributions["total_latency_s"][key]) for key in ("mean", "p50", "p95"))
        lines.append(f"| {model} | {service} | {total} | {_value(distributions['rate_limit_wait_s']['mean'])} | {_value(distributions['input_tokens_actual']['mean'])}/{_value(distributions['completion_tokens_actual']['mean'])} | {stats['transport_error_count']}/{stats['schema_error_count']} | {stats['retry_count']} |")
    lines.extend(["", f"## 同case配对：相对 {reference}", "", "| 模型 | 仅该模型合理 | 仅Qwen合理 | 全量净增 | Holdout净增 |", "|---|---|---|---|---|"])
    for model, comparisons_by_split in comparisons.items():
        pair, holdout = comparisons_by_split["all"]["all"], comparisons_by_split["holdout"]["all"]
        lines.append(f"| {model} | {pair['model_only_accepted']} | {pair['reference_only_accepted']} | {pair['net_model_gain']:+d}/{pair['denominator']} | {holdout['net_model_gain']:+d}/{holdout['denominator']} |")
    lines.extend(["", "## 解释边界", "", *[f"- {note}" for note in result["notes"]], "", "完整dev/holdout、严格/滞后交叉分层、候选召回、禁止动作暴露率、错误详情与配对case列表见comparison.json。"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("comparison.json", "report.md")):
        raise FileExistsError("Use a fresh report output directory; existing results are not overwritten")
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PREDICTIONS_JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        label, separator, path = value.partition("=")
        if not separator or not label.strip() or not path or label in runs:
            parser.error("Each --run must have a distinct nonempty LABEL=PREDICTIONS_JSONL")
        runs[label] = Path(path)
    result = generate(args.inputs, runs, args.output_dir)
    print(json.dumps({"models": list(runs), "case_count_per_model": result["case_count_per_model"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
