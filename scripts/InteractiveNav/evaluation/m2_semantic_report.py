"""Report frozen-label semantic-prompt repeats and separate strategy probes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from scripts.InteractiveNav.evaluation import m2_context_report as context
from scripts.InteractiveNav.evaluation import m2_replay_eval as replay
from scripts.InteractiveNav.evaluation import m2_semantic_checks as semantic_checks


SPLITS = ("all", "dev", "holdout")
ALIGNMENTS = ("all", "strict_version_aligned", "lagged_or_incomplete")
NOTES = [
    "Historical scores are fixed-label public-policy top-1 acceptance, not navigation SR or physical task success.",
    "The historical holdout has already been inspected: this is a regression split, not a fresh blind holdout.",
    "Repeated predictions are pooled as (base_case_id, repeat), not independent new examples; two repeats of 130 eligible cases have denominator 260 (dev 76, holdout 184). Individual rounds are also shown.",
    "Missing predictions, transport errors and invalid complete response schemas stay failures in fixed applicable denominators; unknown secondary preferences are never counted as passes.",
    "Historical secondary checks and synthetic probes are scored separately from the old primary labels and from each other. Synthetic probes are controlled instruction diagnostics, not sampled navigation episodes.",
    "Secondary checks are frozen soft-policy preferences. Report each applicable denominator, pass/fail/unknown, and coverage; zero support cannot establish an effect. Stage-contract conflict is diagnostic only, not correctness.",
    "S1/S2 and combined suggestions alter evidence presentation as well as instructions; they are not pure prompt ablations. S2 may include timestamp-bounded, frame-verified coordinate associations only where available.",
    "Reconstructed exact-version-aligned snapshots are not exact recorded HTTP requests; lagged/incomplete snapshots and partially recovered online eligibility remain limitations.",
    "Top-1 agreement across valid repeats is conditional stability, not accuracy. Missing/invalid repeats are reported separately and are not counted as stable identical empty answers.",
    "Input context tokens use the final attempt's usage; total token consumption and total latency include retries. Repeated decisions within episodes are dependent; no independent-sample significance claim is made.",
]


def _key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["repeat"]), str(row["base_case_id"])


def summarize_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({item["name"] for row in rows for item in row["strategy_checks"]})
    result = {}
    for name in names:
        items = [item for row in rows for item in row["strategy_checks"] if item["name"] == name]
        counts = Counter(item["status"] for item in items)
        applicable = sum(bool(item["applicable"]) for item in items)
        diagnostic = any(item["mode"] == "diagnostic" for item in items)
        pairs = Counter()
        for item in items:
            pairs.update(item.get("pair_counts") or {})
        result[name] = {
            "mode": "diagnostic" if diagnostic else next((item["mode"] for item in items), None),
            "cases": len(items), "applicable": applicable,
            "coverage": context._ratio(applicable, len(items)),
            "pass": counts["pass"], "fail": counts["fail"], "unknown": counts["unknown"],
            "not_applicable": counts["not_applicable"], "diagnostic_only": counts["diagnostic_only"],
            "pass_over_applicable": None if diagnostic else context._ratio(counts["pass"], applicable),
            "known_only_pass": None if diagnostic else context._ratio(counts["pass"], counts["pass"] + counts["fail"]),
            "pair_counts": {status: pairs[status] for status in ("pass", "fail", "unknown")},
            "pair_counts_scope": "valid responses only; invalid response case failures have no resolved pair ranks",
            "conflict_present": sum(bool(item.get("conflict_present")) for item in items),
            "selected_conflicting_hint": sum(bool(item.get("selected_conflicting_hint")) for item in items),
            "valid_applicable_responses": sum(row["valid"] for row in rows if any(
                item["name"] == name and item["applicable"] for item in row["strategy_checks"])),
        }
    return result


def _operational(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = context.summarize(rows)
    return {key: stats[key] for key in (
        "cases", "predictions_received", "missing_predictions", "schema_valid", "schema_error_count",
        "transport_error_count", "retry_count", "prompt_tokens_all_attempts", "total_tokens_all_attempts",
        "distributions", "error_counts",
    )}


def _summary(rows: list[dict[str, Any]], *, historical: bool) -> dict[str, Any]:
    result = context.summarize(rows) if historical else _operational(rows)
    result["strategy_checks"] = summarize_checks(rows)
    return result


def paired_primary(target: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    left = {_key(row): row for row in target if row["main_score_eligible"]}
    right = {_key(row): row for row in control if row["main_score_eligible"]}
    if set(left) != set(right):
        raise ValueError("Paired arms do not have identical fixed eligible case/repeat sets")
    wins = sorted(key for key in left if left[key]["accepted"] and not right[key]["accepted"])
    losses = sorted(key for key in left if right[key]["accepted"] and not left[key]["accepted"])
    return {
        "denominator": len(left), "target_only_accepted": len(wins), "control_only_accepted": len(losses),
        "both_accepted": sum(left[key]["accepted"] and right[key]["accepted"] for key in left),
        "neither_accepted": sum(not left[key]["accepted"] and not right[key]["accepted"] for key in left),
        "net_target_gain": len(wins) - len(losses),
        "net_target_gain_rate": (len(wins) - len(losses)) / len(left) if left else None,
        "target_win_cases": [{"repeat": repeat, "base_case_id": case_id} for repeat, case_id in wins],
        "target_loss_cases": [{"repeat": repeat, "base_case_id": case_id} for repeat, case_id in losses],
    }


def paired_checks(target: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    left, right = {_key(row): row for row in target}, {_key(row): row for row in control}
    if set(left) != set(right):
        raise ValueError("Paired strategy arms differ in case/repeat membership")
    transitions: dict[str, Counter] = defaultdict(Counter)
    modes = {}
    for key in left:
        other = {item["name"]: item for item in right[key]["strategy_checks"]}
        for item in left[key]["strategy_checks"]:
            peer = other[item["name"]]
            if item["applicable"] != peer["applicable"] or item["mode"] != peer["mode"]:
                raise ValueError("Frozen check applicability changed between arms")
            modes[item["name"]] = item["mode"]
            if item["applicable"]:
                transitions[item["name"]][(peer["status"], item["status"])] += 1
    result = {}
    for name, mode in sorted(modes.items()):
        counts = transitions[name]
        wins = sum(n for (before, after), n in counts.items() if before != "pass" and after == "pass")
        losses = sum(n for (before, after), n in counts.items() if before == "pass" and after != "pass")
        result[name] = {
            "mode": mode, "applicable_pairs": sum(counts.values()),
            "transitions_control_to_target": {f"{before}->{after}": n for (before, after), n in sorted(counts.items())},
            "target_only_pass": wins if mode != "diagnostic" else None,
            "control_only_pass": losses if mode != "diagnostic" else None,
            "net_pass_gain": wins - losses if mode != "diagnostic" else None,
        }
    return result


def repeat_stability(rows: list[dict[str, Any]], repeats: list[int]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["base_case_id"]].append(row)
    complete = [group for group in grouped.values() if {row["repeat"] for row in group} == set(repeats)]
    valid = [group for group in complete if all(row["valid"] for row in group)]
    eligible = [group for group in complete if all(row["main_score_eligible"] for row in group)]
    enough = len(repeats) >= 2
    return {
        "repeats": repeats, "cases": len(grouped), "complete_repeat_groups": len(complete),
        "all_repeats_valid": len(valid), "any_missing_or_invalid_repeat": len(grouped) - len(valid),
        "top1_agreement_given_all_valid": context._ratio(sum(len({row["top1"] for row in group}) == 1 for group in valid), len(valid)) if enough else None,
        "ranked_ids_agreement_given_all_valid": context._ratio(sum(len({tuple(row["ranked_ids"]) for row in group}) == 1 for group in valid), len(valid)) if enough else None,
        "primary_all_repeats_accepted": context._ratio(sum(all(row["accepted"] for row in group) for group in eligible), len(eligible)),
        "primary_at_least_one_repeat_accepted": context._ratio(sum(any(row["accepted"] for row in group) for group in eligible), len(eligible)),
        "primary_mixed_acceptance": sum(len({row["accepted"] for row in group}) > 1 for group in eligible),
    }


def build_report(
    cases: list[dict[str, Any]], predictions: list[dict[str, Any]], annotations: list[dict[str, Any]],
    historical_checks: list[dict[str, Any]], synthetic_checks: list[dict[str, Any]],
    *, input_manifest: dict[str, Any] | None = None, run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure scoring entry point; consume frozen checks, never rebuild the rubric."""
    case_index = context._index_unique(cases, "case_id")
    prediction_index = context._index_unique(predictions, "case_id")
    labels = context._index_unique(annotations, "case_id")
    checks = {"historical": context._index_unique(historical_checks, "case_id"),
              "synthetic": context._index_unique(synthetic_checks, "case_id")}
    if set(prediction_index) - set(case_index):
        raise ValueError("Predictions contain case IDs outside the frozen inputs")
    if not cases:
        raise ValueError("No frozen cases")
    arms = list((input_manifest or {}).get("arms") or sorted({case["arm"] for case in cases}))
    repeats = sorted({int(case["repeat"]) for case in cases})
    if set(arms) != {case["arm"] for case in cases} or len(arms) != len(set(arms)):
        raise ValueError("Frozen arm manifest differs from cases")
    if input_manifest and "repeats" in input_manifest and repeats != list(range(1, input_manifest["repeats"] + 1)):
        raise ValueError("Frozen repeat manifest differs from cases")
    cells: dict[tuple[str, int], set] = defaultdict(set)
    fixed_candidates = {}
    rows = []
    for case in cases:
        dataset, base, repeat = case["dataset"], case["base_case_id"], int(case["repeat"])
        if dataset not in checks:
            raise ValueError(f"Unknown dataset: {dataset}")
        identity = (dataset, base)
        cell = cells[(case["arm"], repeat)]
        if identity in cell:
            raise ValueError("Duplicate frozen arm/repeat/base case")
        cell.add(identity)
        candidates = case["request"]["candidates"]
        if identity in fixed_candidates and fixed_candidates[identity] != candidates:
            raise ValueError("Prompt arms changed candidate facts, membership or order")
        fixed_candidates[identity] = candidates
        annotation = labels[base] if dataset == "historical" else {"split": "synthetic", "main_score_eligible": False}
        check = checks[dataset][case["source_case_id"]]
        if set(check["candidate_ids"]) != {candidate["id"] for candidate in candidates}:
            raise ValueError("Frozen checks do not match candidate membership")
        names = [item["name"] for item in check["checks"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate strategy check name")
        prediction = prediction_index.get(case["case_id"])
        for field in ("base_case_id", "source_case_id", "arm", "repeat", "dataset"):
            if prediction and field in prediction and prediction[field] != case[field]:
                raise ValueError(f"Prediction metadata differs from frozen input: {field}")
        row = context._score_case(case, prediction, annotation)
        response = replay.validate_response(prediction["response"], check["candidate_ids"])[0] if row["valid"] else None
        evaluated = semantic_checks.score_checks(check, response)
        mode_by_name = {item["name"]: item["mode"] for item in check["checks"]}
        row.update(dataset=dataset, repeat=repeat, source_case_id=case["source_case_id"],
                   ranked_ids=response["ranked_ids"] if response else [],
                   strategy_checks=[dict(item, mode=mode_by_name[item["name"]]) for item in evaluated["checks"]])
        rows.append(row)
    expected = set(fixed_candidates)
    if any(cells[(arm, repeat)] != expected for arm in arms for repeat in repeats):
        raise ValueError("Incomplete frozen arm/repeat case matrix")
    for dataset in checks:
        if set(checks[dataset]) != {case["source_case_id"] for case in cases if case["dataset"] == dataset}:
            raise ValueError("Frozen check set differs from source cases")
    by_arm = {}
    repeat_keys = [str(repeat) for repeat in repeats] + ["all"]
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        summaries = {}
        for repeat in repeat_keys:
            repeated = [row for row in selected if repeat == "all" or row["repeat"] == int(repeat)]
            historical = [row for row in repeated if row["dataset"] == "historical"]
            summaries[repeat] = {
                "historical": {split: {alignment: _summary(context._subset(historical, split, alignment), historical=True)
                                         for alignment in ALIGNMENTS} for split in SPLITS},
                "synthetic": _summary([row for row in repeated if row["dataset"] == "synthetic"], historical=False),
                "operational_all_datasets": _operational(repeated),
            }
        stability = {dataset: repeat_stability([row for row in selected if row["dataset"] == dataset], repeats)
                     for dataset in ("historical", "synthetic")}
        for key in ("primary_all_repeats_accepted", "primary_at_least_one_repeat_accepted", "primary_mixed_acceptance"):
            stability["synthetic"].pop(key)
        by_arm[arm] = {"by_repeat": summaries, "stability": stability}
    comparisons = []
    for arm in arms:
        controls = (["b"] if arm != "b" and "b" in arms else [])
        if arm in {"p2", "p3", "p4", "p5"} and "p1" in arms:
            controls.append("p1")
        if arm.startswith("s") and "p6" in arms:
            controls.append("p6")
        for control in controls:
            paired = {}
            for repeat in repeat_keys:
                target_rows = [row for row in rows if row["arm"] == arm and (repeat == "all" or row["repeat"] == int(repeat))]
                control_rows = [row for row in rows if row["arm"] == control and (repeat == "all" or row["repeat"] == int(repeat))]
                historical_target = [row for row in target_rows if row["dataset"] == "historical"]
                historical_control = [row for row in control_rows if row["dataset"] == "historical"]
                paired[repeat] = {
                    "historical_primary": {split: {alignment: paired_primary(
                        context._subset(historical_target, split, alignment), context._subset(historical_control, split, alignment))
                        for alignment in ALIGNMENTS} for split in SPLITS},
                    "strategy_checks": {dataset: paired_checks(
                        [row for row in target_rows if row["dataset"] == dataset],
                        [row for row in control_rows if row["dataset"] == dataset]) for dataset in ("historical", "synthetic")},
                }
            comparisons.append({"target": arm, "control": control, "by_repeat": paired})
    support = {dataset: {"unique_cases": len(source), "applicable_per_round": {
        name: sum(item["applicable"] for row in source.values() for item in row["checks"] if item["name"] == name)
        for name in sorted({item["name"] for row in source.values() for item in row["checks"]})},
        "diagnostic_checks": sorted({item["name"] for row in source.values() for item in row["checks"] if item["mode"] == "diagnostic"})}
        for dataset, source in checks.items()}
    return {
        "schema_version": "m2_semantic_report_v1", "metric": "fixed_historical_public_top1_acceptance",
        "input_manifest": input_manifest or {}, "run_manifest": run_manifest or {},
        "planned_requests": len(cases), "predictions_received": len(predictions),
        "arms": arms, "repeats": repeats, "check_support": support,
        "by_arm": by_arm, "paired_comparisons": comparisons, "case_scores": rows, "limitations": NOTES,
    }


def _fraction(metric: dict[str, Any] | None) -> str:
    return context._fraction(metric) if metric is not None else "不计分"


def _number(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# M2 语义提示词与建议实验", "", "历史指标是固定标签的合理 top-1 接受率，不是导航成功率。旧 holdout 已检查过，仅作为回归集。历史与 synthetic 独立计分；pooled 分母按 case × repeat 合并，不代表新增独立场景。", "",
             "## 历史主指标（单轮与合并）", "", "| Arm | 轮次 | 主指标 | Dev | Holdout 回归 | 严格版本对齐 | 滞后/不完整 |", "|---|---|---|---|---|---|---|"]
    for arm in report["arms"]:
        for repeat, summary in report["by_arm"][arm]["by_repeat"].items():
            stats = summary["historical"]
            values = [_fraction(stats[split]["all"]["main_acceptable_top1"]) for split in SPLITS]
            values += [_fraction(stats["all"][alignment]["main_acceptable_top1"]) for alignment in ALIGNMENTS[1:]]
            lines.append(f"| {arm} | {repeat} | " + " | ".join(values) + " |")
    lines += ["", "## 配对变化（目标 arm 相对对照，合并两轮）", "", "| 目标 | 对照 | 仅目标接受 | 仅对照接受 | 净增/配对分母 | Dev 净增 | Holdout 净增 |", "|---|---|---|---|---|---|---|"]
    for comparison in report["paired_comparisons"]:
        stats = comparison["by_repeat"]["all"]["historical_primary"]
        pooled = stats["all"]["all"]
        lines.append(f"| {comparison['target']} | {comparison['control']} | {pooled['target_only_accepted']} | {pooled['control_only_accepted']} | {pooled['net_target_gain']}/{pooled['denominator']} | {stats['dev']['all']['net_target_gain']} | {stats['holdout']['all']['net_target_gain']} |")
    for dataset, title in (("historical", "历史策略检查"), ("synthetic", "Synthetic 独立策略检查")):
        support = report["check_support"][dataset]
        lines += ["", f"## {title}", "", f"单轮样本数 {support['unique_cases']}。各检查只使用自己的适用分母；pass/applicable 是把 unknown 保留在分母内的下界。阶段冲突只诊断，不评正确率。", "",
                  "| Arm | 检查 | 适用/总数 | Pass | Fail | Unknown | Pass/适用 |", "|---|---|---|---|---|---|---|"]
        for arm in report["arms"]:
            summary = report["by_arm"][arm]["by_repeat"]["all"][dataset]
            if dataset == "historical":
                summary = summary["all"]["all"]
            for name, stat in summary["strategy_checks"].items():
                details = f"诊断 {stat['diagnostic_only']}；选冲突 hint {stat['selected_conflicting_hint']}" if stat["mode"] == "diagnostic" else _fraction(stat["pass_over_applicable"])
                lines.append(f"| {arm} | {name} | {stat['applicable']}/{stat['cases']} | {stat['pass']} | {stat['fail']} | {stat['unknown']} | {details} |")
    lines += ["", "## 请求、token 与耗时（历史和 synthetic 请求合并）", "", "| Arm | 有效 schema | schema/request/缺失 | 重试 | 输入 token mean/p95 | 总 token | 含重试耗时 mean/p95 秒 |", "|---|---|---|---|---|---|---|"]
    for arm in report["arms"]:
        stat = report["by_arm"][arm]["by_repeat"]["all"]["operational_all_datasets"]
        tokens, latency = stat["distributions"]["input_tokens_actual"], stat["distributions"]["total_latency_s"]
        lines.append(f"| {arm} | {_fraction(stat['schema_valid'])} | {stat['schema_error_count']}/{stat['transport_error_count']}/{stat['missing_predictions']} | {stat['retry_count']} | {_number(tokens['mean'])}/{_number(tokens['p95'])} | {stat['total_tokens_all_attempts']} | {_number(latency['mean'])}/{_number(latency['p95'])} |")
    lines += ["", "## 重复稳定性", "", "| Arm | 数据集 | 两轮均有效 | 有效条件 top-1 一致 | 至少一次无效/缺失 | 主标签两轮均接受 | 主标签至少一轮接受 |", "|---|---|---|---|---|---|---|"]
    for arm in report["arms"]:
        for dataset, stat in report["by_arm"][arm]["stability"].items():
            lines.append(f"| {arm} | {dataset} | {stat['all_repeats_valid']}/{stat['cases']} | {_fraction(stat['top1_agreement_given_all_valid'])} | {stat['any_missing_or_invalid_repeat']} | {_fraction(stat.get('primary_all_repeats_accepted'))} | {_fraction(stat.get('primary_at_least_one_repeat_accepted'))} |")
    lines += ["", "## 解释边界", "", *[f"- {note}" for note in report["limitations"]], "",
              "report.json 保留逐轮/分 split/版本层统计、各检查的适用性与 unknown、配对状态转移、错误类型、token 明细及逐 case 分数。"]
    return "\n".join(lines) + "\n"


def generate(inputs: Path, run: Path, output_dir: Path) -> dict[str, Any]:
    if any((output_dir / name).exists() for name in ("report.json", "report.md")):
        raise FileExistsError("Report exists; use a fresh output directory")
    manifest = json.loads((inputs / "manifest.json").read_text())
    run_manifest = json.loads((run / "manifest.json").read_text())
    for name, expected in manifest.get("generated_artifact_sha256", {}).items():
        if context._sha(inputs / name) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    if manifest.get("cases_file_sha256") and context._sha(inputs / "cases.jsonl") != manifest["cases_file_sha256"]:
        raise ValueError("Frozen cases file hash mismatch")
    if run_manifest.get("inputs_manifest_sha256") and context._sha(inputs / "manifest.json") != run_manifest["inputs_manifest_sha256"]:
        raise ValueError("Run uses a different frozen input manifest")
    paths = {"cases": inputs / "cases.jsonl", "predictions": run / "predictions.jsonl",
             "annotations": inputs / "base_annotations.jsonl", "historical_checks": inputs / "checks.jsonl",
             "synthetic_checks": inputs / "synthetic_checks.jsonl"}
    rows = {key: context._rows(path) if path.exists() else [] for key, path in paths.items()}
    if any(not path.exists() for key, path in paths.items() if key != "predictions"):
        raise FileNotFoundError("Missing frozen input/label/check artifact")
    report = build_report(**rows, input_manifest=manifest, run_manifest=run_manifest)
    report["file_sha256"] = {key: context._sha(path) if path.exists() else None for key, path in paths.items()}
    report["file_sha256"]["report_generator"] = context._sha(Path(__file__))
    report["file_sha256"]["context_scorer"] = context._sha(Path(context.__file__))
    report["file_sha256"]["semantic_scorer"] = context._sha(Path(semantic_checks.__file__))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "run", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = generate(args.inputs, args.run, args.output_dir)
    print(json.dumps({"planned_requests": report["planned_requests"], "predictions_received": report["predictions_received"],
                      "report": str(args.output_dir / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
