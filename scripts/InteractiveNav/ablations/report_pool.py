"""Summarize saved evaluator facts from a shared ablation pool (no rescoring)."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


NAMES = {"historical_full": "历史 Full（仅参考）", "no_interaction_graph": "Flat Object Memory",
         "no_task_decision": "Greedy Selection", "no_outcome_update": "Perception-only Update"}
FIELDS = ["nav_success", "success", "task_success", "required_interaction_success", "spl",
          "interaction_precision_episode", "episode_total_cost", "navigation_path_length_m",
          "interaction_action_count", "valid_interaction_attempt_count", "error_interaction_attempt_count",
          "repeated_interaction_attempt_count", "task_irrelevant_interaction_attempt_count",
          "failed_interaction_attempt_count", "step_count", "episode_step_budget", "elapsed_seconds",
          "terminal_reason", "scoring_eligible"]


def metrics(path):
    counts, errors = Counter(), Counter()
    tokens = latency = 0
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            role = row.get("role", "unknown")
            counts[role] += 1
            if row.get("error"):
                errors[role] += 1
            tokens += row.get("total_tokens") or 0
            latency += row.get("latency_s") or 0
    return {"roles": dict(counts), "errors": dict(errors), "tokens": tokens, "latency_sum_s": latency}


def make_row(variant, batch_row, result):
    row = {"variant": variant, "episode_index": batch_row["episode_index"],
           "scene_index": batch_row["episode_index"] - 2000, "completed": batch_row.get("completed", False),
           "runner_exit_code": batch_row.get("runner_exit_code"), **{k: result.get(k) for k in FIELDS}}
    row["mllm"] = metrics(Path(batch_row.get("attempt_dir", "/nonexistent")) / "mllm_metrics.jsonl")
    row["result_path"] = batch_row.get("episode_result_path")
    row["paper_metric_config"] = result.get("paper_metric_config")
    return row


def aggregate(rows):
    eligible = [r for r in rows if r["completed"] and r["scoring_eligible"] is True]
    result = {"reported": len(rows), "completed": sum(bool(r["completed"]) for r in rows), "eligible": len(eligible)}
    for key in FIELDS:
        values = [r[key] for r in eligible if isinstance(r[key], (int, float))]
        if values:
            result[key] = statistics.mean(values)
            result[key + "_sum"] = sum(values)
    result["termination_counts"] = dict(Counter(r["terminal_reason"] for r in rows))
    result["mllm_roles"] = dict(sum((Counter(r["mllm"]["roles"]) for r in rows), Counter()))
    result["mllm_errors"] = dict(sum((Counter(r["mllm"]["errors"]) for r in rows), Counter()))
    result["mllm_tokens"] = sum(r["mllm"]["tokens"] for r in rows)
    result["mllm_latency_sum_s"] = sum(r["mllm"]["latency_sum_s"] for r in rows)
    return result


def summarize(root):
    manifest = json.loads((root / "pool_manifest.json").read_text())
    rows = [make_row("historical_full", saved, saved)
            for saved in manifest["selected_full_rows"]]
    variants = list(dict.fromkeys(variant for variant, _ in manifest["jobs"]))
    for variant in variants:
        for path in sorted((root / variant).glob("episode_*/batch_task_summary.json")):
            batch_row = json.loads(path.read_text())
            result_path = Path(batch_row.get("episode_result_path") or "/nonexistent")
            result = json.loads(result_path.read_text()).get("result", {}) if result_path.is_file() else {}
            rows.append(make_row(variant, batch_row, result))
    displayed_variants = ["historical_full", *variants]
    groups = {v: aggregate([r for r in rows if r["variant"] == v]) for v in displayed_variants}
    planned = Counter(variant for variant, _ in manifest["jobs"])
    complete = all(groups[v]["reported"] == planned[v] for v in variants)
    report = {"complete": complete, "groups": groups, "episodes": rows, "paired_common_success": {}}
    full = {r["episode_index"]: r for r in rows if r["variant"] == "historical_full"}
    for variant in variants:
        paired = [r for r in rows if r["variant"] == variant and r["completed"] and r["scoring_eligible"]
                  and r["nav_success"] and r["episode_index"] in full
                  and full[r["episode_index"]]["nav_success"]]
        report["paired_common_success"][variant] = {
            "indices": [r["episode_index"] for r in paired],
            "historical_full": aggregate([full[r["episode_index"]] for r in paired]),
            "ablation": aggregate(paired),
        }
    (root / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "scene_index", "episode_index", "completed", *FIELDS])
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in writer.fieldnames} for row in rows)
    def fmt(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"
    def pct(value):
        return "—" if value is None else f"{100*value:.1f}%"
    lines = ["# Mixed 定向子集消融结果", "",
             "状态：" + (f"{len(manifest['jobs'])} 项已报告。" if complete else "运行中，以下为部分结果。"), "",
             "场景：" + "、".join(str(i-2000) for i in manifest["config"]["episode_indices"]) + "。",
             f"按历史 Full 成功/有效交互筛选，不代表完整 mixed 的无偏平均。只启动 {len(variants)} 组消融；历史 Full 不重跑。",
             f"消融共同使用 {manifest.get('m1_refresh_profile', 'baseline')} M1 profile；历史 Full 可能采用不同配置及并发，仅参考。",
             f"{manifest['config']['workers']} 个共享 worker，{manifest['config']['step_budget_mode']} 预算 max {manifest['config']['max_steps']}，"
             f"{'录制' if manifest['config'].get('recording', True) else '无录制'}。Paper SR=nav_success；ICS=result.success（含交互条件）。", "",
             "| 组 | 已报告/有效 | Paper SR | ICS | ISR | SPL | IP | 平均 Cost | 平均路径 m | 交互总数 | 重复总数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in displayed_variants:
        group = groups[variant]
        lines.append(f"| {NAMES.get(variant, variant)} | {group['reported']}/{group['eligible']} | {pct(group.get('nav_success'))} | {pct(group.get('success'))} | {pct(group.get('required_interaction_success'))} | {fmt(group.get('spl'), 3)} | {pct(group.get('interaction_precision_episode'))} | {fmt(group.get('episode_total_cost'))} | {fmt(group.get('navigation_path_length_m'))} | {fmt(group.get('interaction_action_count_sum'), 0)} | {fmt(group.get('repeated_interaction_attempt_count_sum'), 0)} |")
    lines += ["", "IP 为逐场宏平均；Cost 使用 evaluator 保存的 episode_total_cost，未重新定义成本或成功条件。",
              "完成但 scoring_eligible=false 的任务不进入指标分母；运行错误不伪装成正常导航失败。", "",
              "## 模型调用", "", "| 组 | 对象 M1 | 房间推理 | M2 | 其他 | 错误请求 | 总 tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in displayed_variants:
        g = groups[variant]
        c = g["mllm_roles"]
        lines.append(f"| {NAMES.get(variant, variant)} | {c.get('attribute_inference', 0)} | {c.get('room_attribute_inference', 0)} | {c.get('subgoal_selection', 0)} | {sum(v for k,v in c.items() if k not in {'attribute_inference','room_attribute_inference','subgoal_selection'})} | {sum(g['mllm_errors'].values())} | {g['mllm_tokens']} |")
    lines += ["", "## 逐场结果", "", "每格：Paper SR / ICS / ISR；有效交互/总交互；终止原因。", "",
              "| 场景 | " + " | ".join(NAMES.get(v, v) for v in displayed_variants) + " |",
              "|---|" + "---|" * len(displayed_variants)]
    lookup = {(r["variant"], r["episode_index"]): r for r in rows}
    for index in manifest["config"]["episode_indices"]:
        cells = []
        for variant in displayed_variants:
            r = lookup.get((variant, index))
            if r is None:
                cells.append("运行中/排队")
            elif not r["completed"]:
                cells.append("运行异常：" + str(r["runner_exit_code"]))
            else:
                cells.append(f"{int(bool(r['nav_success']))}/{int(bool(r['success']))}/{int(bool(r['required_interaction_success']))}; {r['valid_interaction_attempt_count']}/{r['interaction_action_count']}; {r['terminal_reason']}")
        lines.append(f"| {index-2000} | " + " | ".join(cells) + " |")
    lines += ["", "## 共同成功场景的成本", "", "以下仍为与历史 Full 的探索性对照，不能消除配置与随机性混杂。"]
    for variant, pair in report["paired_common_success"].items():
        lines.append(f"- {NAMES.get(variant, variant)}：n={len(pair['indices'])}；历史 Full / 消融平均 Cost：{fmt(pair['historical_full'].get('episode_total_cost'))} / {fmt(pair['ablation'].get('episode_total_cost'))}。")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    report = summarize(parser.parse_args().output_dir)
    print(json.dumps({"complete": report["complete"], "groups": report["groups"]}, ensure_ascii=False))
