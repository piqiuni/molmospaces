#!/usr/bin/env python3
"""Audit mixed episodes and report explicit quality-filtered scoring subsets.

This does not repair a policy, replay physics, or delete benchmark scenes.
Post-hoc subsets are diagnostics, not replacements for full-benchmark scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.InteractiveNav.rescore_benchmark_goals import aggregate, read_json, sha256

AUDIT_VERSION = "mixed_quality_audit_v1"
LOG_PATTERNS = {
    "active_make_plan_rejected": "move_base must be in an inactive state to make a plan",
    "actionlib_preempt_done_race": "Received comm state PREEMPTING when in simple state DONE",
    "untracked_goal_callback": "Got a transition callback on a goal handle that we're not tracking",
    "traceback": r"Traceback \(most recent call last\)",
    "closed_topic": r"publish\(\) to a closed topic",
    "memory_cleanup_error": r"double free or corruption|munmap_chunk\(\): invalid pointer|free\(\): invalid pointer",
    "connection_refused": "Connection refused",
    "process_died": "process has died",
}


def scan_log(path):
    if not path.is_file():
        return {"present": False, "path": str(path)}
    active, shutdown = Counter(), Counter()
    examples = {}
    closing = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if "killing on exit" in line:
                closing = True
            for name, pattern in LOG_PATTERNS.items():
                if re.search(pattern, line):
                    counts = shutdown if closing else active
                    counts[name] += 1
                    key = ("shutdown:" if closing else "active:") + name
                    examples.setdefault(key, {"line": number, "text": line.strip()[:500]})
    return {"present": True, "path": str(path), "sha256": sha256(path),
            "active_counts": dict(active), "shutdown_counts": dict(shutdown), "examples": examples}


def classify_episode(episode, result, smooth_records, private_visualization):
    """Apply the same predicates to successful and unsuccessful mixed episodes."""
    nav = episode["interactive_nav"]
    scene_reasons, execution_reasons, reviews, evidence = [], [], [], {}
    if result.get("scoring_eligible") is False:
        scene_reasons.extend(result.get("scoring_exclusion_reasons") or ["runtime_ineligible_unspecified"])
        evidence["runtime_gate"] = {"terminal_reason": result.get("terminal_reason"),
                                    "step_count": result.get("step_count"),
                                    "checks": result.get("runtime_consistency", {}).get("checks")}
    attempts = result.get("interaction_attempts", [])
    container_rows = [row for row in nav.get("interactions", [])
                      if str(row.get("type", "")).startswith("container_")
                      and "reveal_target_object" in row.get("effect_types", [])]
    required_ids = set(nav.get("oracle_plan", {}).get("required_interaction_ids", []))
    # A strict original-target success is a counterexample to the benchmark's
    # claimed necessity of opening the container. Relaxed-category successes
    # are deliberately insufficient (e.g. the nearby CD on the dresser top).
    strict_counterexample = (
        set(nav.get("interaction_domains", [])) == {"channel", "container"}
        and nav.get("interaction_requirement") == "required"
        and nav.get("initial_state", {}).get("container_joints_closed") is True
        and any(row["interaction_id"] in required_ids for row in container_rows)
        and result.get("status") == "complete" and result.get("scoring_eligible") is True
        and result.get("policy_name") == "ros_object_goal_rule"
        and result.get("nav_success") is True
        and result.get("terminal_reason") == "target_found"
        and result.get("required_interaction_success") is False
        and bool(attempts)
        and all(attempt.get("node_type") == "portal" for attempt in attempts)
    )
    if strict_counterexample:
        scene_reasons.append("strict_target_reached_without_required_container_interaction")
        evidence["container_necessity_counterexample"] = {
            "target": nav["target"]["selected_instance"],
            "target_distance_m": result.get("target_distance_m"),
            "target_visibility_fraction": result.get("target_visibility_fraction"),
            "interaction_attempt_count": len(attempts), "all_attempts_portal_only": True,
            "frozen_minimal_plan_validation": nav.get("generation_validation", {}).get("minimal_plan_validation"),
            "caveat": "Shows a violated runtime task premise, not whether leakage or uncommanded physics caused it.",
        }
    backend_evidence = []
    for source_path, record in smooth_records:
        private_result = record.get("result", {})
        regions = private_result.get("region_results", [])
        convergence = private_result.get("view_restore_convergence") or {}
        matching = [a for a in attempts if a.get("command_id") == private_result.get("command_id")]
        restore_fault = (
            private_result.get("sequence_type") == "drawer_scan"
            and private_result.get("failure_reason") == "drawer_view_restore_timeout"
            and private_result.get("success") is False
            and bool(regions) and all(r.get("success") is True for r in regions)
            and private_result.get("final_close_success") is True
            and convergence.get("checked") is True and convergence.get("converged") is False
            and any(a.get("success") is False and a.get("failure_reason") == "drawer_scan_execution_failed"
                    for a in matching)
        )
        if restore_fault:
            execution_reasons.append("drawer_view_restore_timeout_after_successful_scan")
            backend_evidence.append({"path": str(source_path), "command_id": private_result.get("command_id"),
                                     "region_count": len(regions), "all_regions_successful": True,
                                     "final_close_success": True, "convergence": convergence,
                                     "restore_steps": private_result.get("view_restore_steps_elapsed"),
                                     "restore_max_steps": private_result.get("view_restore_max_steps"),
                                     "caveat": "This attempt is contaminated; it does not prove the intended target would otherwise succeed."})
    if backend_evidence:
        evidence["execution_restore_faults"] = backend_evidence
    discovery = private_visualization.get("transient_target_discovery")
    if discovery and not result.get("nav_success"):
        reviews.append("private_drawer_target_seen_but_public_goal_not_completed")
        evidence["private_drawer_discovery"] = discovery
        evidence["private_discovery_caveat"] = (
            "Private pixels are not proof of public publication, policy receipt, or a valid goal claim. "
            "Keep this episode; investigate perception filtering/goal completion before assigning causality."
        )
    return {"scene_exclusion_reasons": sorted(set(scene_reasons)),
            "execution_exclusion_reasons": sorted(set(execution_reasons)),
            "review_flags": reviews, "evidence": evidence}


def score_profiles(results, audits):
    scene_excluded = {r["episode_index"] for r in audits if r["scene_exclusion_reasons"]}
    execution_excluded = {r["episode_index"] for r in audits if r["execution_exclusion_reasons"]}
    eligible = [r for r in results if r.get("scoring_eligible") is True]
    scene_valid = [r for r in eligible if r["episode_index"] not in scene_excluded]
    profiles = {"all_planned": results, "original_eligible": eligible,
                "scene_valid": scene_valid,
                "scene_and_execution_valid": [r for r in scene_valid if r["episode_index"] not in execution_excluded]}
    summary = {}
    for name, rows in profiles.items():
        summary[name] = {}
        for group in ("all", "mixed", "channel", "container"):
            subset = [r for r in rows if group == "all"
                      or (group == "mixed" and set(r.get("domains", [])) == {"channel", "container"})
                      or set(r.get("domains", [])) == {group}]
            summary[name][group] = aggregate(subset)
    return summary, profiles["scene_and_execution_valid"]


def run(evaluation, rescore_dir, output):
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    previous = read_json(rescore_dir / "report.json")
    if sha256(evaluation / "summary.json") != previous["source_summary_sha256"]:
        raise ValueError("Evaluation summary changed since goal rescoring")
    benchmark = Path(previous["benchmark"])
    if sha256(benchmark) != previous["benchmark_sha256"]:
        raise ValueError("Benchmark changed since goal rescoring")
    episodes = read_json(benchmark)
    source = read_json(evaluation / "summary.json")
    results = read_json(rescore_dir / "rescored_results.json")
    by_index = {r["episode_index"]: r for r in results}
    old_audits = {r["episode_index"]: r for r in previous["episodes"]}
    if len(by_index) != len(results) or set(by_index) != {r["episode_index"] for r in source["episodes"]}:
        raise ValueError("Rescored result coverage mismatch")
    audits = []
    for row in source["episodes"]:
        index = row["episode_index"]
        if set(episodes[index]["interactive_nav"]["interaction_domains"]) != {"channel", "container"}:
            continue
        path, attempt = Path(row["episode_result_path"]), Path(row["attempt_dir"])
        if not path.resolve().is_relative_to(evaluation.resolve()):
            raise ValueError(f"Episode {index} points outside evaluation directory")
        if sha256(path) != old_audits[index]["source_sha256"]:
            raise ValueError(f"Episode {index} changed since goal rescoring")
        result = read_json(path)["result"]
        if result["case_id"] != episodes[index]["interactive_nav"]["case_id"] or result["case_id"] != by_index[index]["case_id"]:
            raise ValueError(f"Episode {index} identity mismatch")
        sidecar = path.with_name("episode_visualization.json")
        private = read_json(sidecar)
        smooth = [(p, read_json(p)) for p in sorted((attempt / "eval/smooth_interactions").glob("*.json"))]
        audit = classify_episode(episodes[index], result, smooth, private)
        audit.update(episode_index=index, case_id=result["case_id"], house_index=result["house_index"],
                     terminal_reason=result["terminal_reason"], original_nav_success=result["nav_success"],
                     revised_nav_success=by_index[index]["nav_success"],
                     revised_interaction_conditioned_success=by_index[index]["success"],
                     source_result=str(path), source_result_sha256=sha256(path),
                     private_visualization_path=str(sidecar), private_visualization_sha256=sha256(sidecar),
                     smooth_interaction_sources=[{"path": str(p), "sha256": sha256(p)} for p, _ in smooth],
                     log_audit=scan_log(attempt / "roslaunch.log"))
        audits.append(audit)
    aggregates, retained = score_profiles(results, audits)
    exclusions = [{"episode_index": r["episode_index"], "case_id": r["case_id"],
                   "scene_reasons": r["scene_exclusion_reasons"], "execution_reasons": r["execution_exclusion_reasons"],
                   "scope": "this_evaluation_attempt_only; revalidate_or_rerun_before_reuse"}
                  for r in audits if r["scene_exclusion_reasons"] or r["execution_exclusion_reasons"]]
    log_counts = Counter()
    log_affected = {}
    for audit in audits:
        for name, count in audit["log_audit"].get("active_counts", {}).items():
            log_counts[name] += count
            log_affected.setdefault(name, []).append(audit["episode_index"])
    report = {"schema_version": AUDIT_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
              "source_goal_rescore": str(rescore_dir), "source_goal_report_sha256": sha256(rescore_dir / "report.json"),
              "source_rescored_results_sha256": sha256(rescore_dir / "rescored_results.json"),
              "implementation_sha256": sha256(Path(__file__)), "benchmark_sha256": sha256(benchmark),
              "policy": "Same predicates for every mixed episode, including successes. Unresolved warnings do not justify exclusions.",
              "caveat": "Post-hoc conditional quality subsets; not improved algorithm performance or an unbiased replacement benchmark.",
              "aggregates": aggregates, "exclusions": exclusions,
              "active_log_counts": dict(log_counts), "active_log_affected_episodes": log_affected, "episodes": audits}
    lines = ["# Mixed 任务质量审计与剔除后重评分", "",
             "沿用 goal_equivalence_v1；审计所有 mixed 场景（含成功），原始结果、benchmark 不修改。",
             "这是事后质量条件子集，不是算法性能提升，也不是新的无偏全量 benchmark 成绩。", "",
             "## 分母和评分", "",
             "scene_valid 排除已有 runtime-ineligible 与严格原目标未开容器即成功的必要性反例；",
             "scene_and_execution_valid 再排除已核实的抽屉扫描后姿态恢复超时。", "",
             "| 范围/口径 | N | 目标成功 | 完整交互任务成功 | 原参考 SPL | Total Cost |",
             "|---|---:|---:|---:|---:|---:|"]
    for group in ("mixed", "all"):
        for profile, groups in aggregates.items():
            r = groups[group]
            def fmt(value):
                return "N/A" if value is None else f"{value:.4f}"
            nav_rate = "N/A" if r["nav_sr"] is None else f"{r['nav_sr']:.2%}"
            formal_rate = "N/A" if r["interaction_conditioned_sr"] is None else f"{r['interaction_conditioned_sr']:.2%}"
            lines.append(f"| {group}/{profile} | {r['episodes']} | {r['nav_successes']} ({nav_rate}) | "
                         f"{r['interaction_conditioned_successes']} ({formal_rate}) | {fmt(r['spl_original_reference'])} | {fmt(r['mean_total_cost'])} |")
    lines.extend(["", "SPL 沿用原目标参考路径，未重算等价目标最短路。交互事实和成功标签不再修改，仅改变计分集合。", "",
                  "## 剔除清单", ""])
    for row in exclusions:
        lines.append(f"- **{row['episode_index']}**：" + "; ".join(row["scene_reasons"] + row["execution_reasons"]))
    lines.extend(["", "2014 原本目标成功，剔除它也会减少成功分子；不是只剔除失败。",
                  "执行器超时说明该次运行受污染，不证明其原目标本来必然成功，也不永久删除这个 benchmark case。", "",
                  "## 未据此剔除的问题", ""])
    for name, indices in log_affected.items():
        lines.append(f"- `{name}`：运行期 {log_counts[name]} 次，涉及 {len(indices)} 场：{indices}。")
    lines.extend(["- move_base/make_plan 与 actionlib 竞态也出现在成功场景；仅凭报错不能挑选失败场景剔除。",
                  "- 清理阶段 closed-topic、内存析构错误单独记录，不作为先前任务失败原因。"])
    for row in audits:
        if row["review_flags"]:
            discovery = row["evidence"].get("private_drawer_discovery", {})
            lines.append(f"- {row['episode_index']}：抽屉中私有 GT 曾见目标（{discovery.get('visible_pixels')} pixels，"
                         f"距离 {discovery.get('distance_m', 0):.3f} m），但没有公开完成证据；保留原判定，待查感知过滤/完成链路。")
    lines.extend(["", "完整逐场证据、来源 SHA256、运行期/关闭期日志行号见 report.json。", ""])
    output.mkdir(parents=True)
    for name, value in (("report.json", report), ("exclusions.json", exclusions), ("filtered_results.json", retained)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--goal-rescore-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.evaluation_dir.resolve(), args.goal_rescore_dir.resolve(), args.output_dir.resolve())
    print(json.dumps({"report": str(args.output_dir / "report.md"), "exclusions": report["exclusions"],
                      "scores": report["aggregates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
