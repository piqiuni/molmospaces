#!/usr/bin/env python3
"""Rescore completed V3 runs without replaying simulation or overwriting evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.InteractiveNav.evaluation.goal_equivalence import (
    PROTOCOL, equivalent_target, rescore_result,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def door_alias(opaque_id):
    digest = hashlib.blake2s(opaque_id.encode("utf-8"), digest_size=4).digest()
    return f"door_{int.from_bytes(digest, 'big') % 9999 + 1:04d}"


def reconstruct_v17_registry(episode, metadata, xml_path, eval_log):
    """Read-only reconstruction of v17's sorted registration domain.

    Guard the legacy path with XML body presence and the *recorded* complete
    channel alias set. Never interpret an ordinal as an object-list index alone.
    Future releases should persist a private registry instead of reconstructing.
    """
    modifications = episode.get("scene_modifications", {})
    if modifications.get("added_objects") or modifications.get("removed_objects"):
        raise ValueError("legacy_registry_scene_additions_or_removals")
    xml = ET.parse(xml_path).getroot()
    if xml.findall(".//include"):
        raise ValueError("legacy_registry_unexpanded_xml")
    bodies = {node.get("name"): node for node in xml.findall(".//body") if node.get("name")}
    objects = metadata["objects"]
    top_names = {node.get("name") for node in xml.findall("./worldbody/body")}
    extras = {name for name in top_names - objects.keys()
              if name and not re.match(r"^(room|wall|ceiling|floor|robot)(_|$)", name)}
    if extras:
        raise ValueError("legacy_registry_unmodelled_top_level_bodies: " + ",".join(sorted(extras)))
    sources = set(objects) & bodies.keys()
    channel_sources = set()
    for root_name, item in objects.items():
        if "doorway" not in root_name:
            continue
        leaves = []
        for name, original_name in item.get("name_map", {}).get("bodies", {}).items():
            body = bodies.get(name)
            if "_door_" in original_name and body is not None and body.findall(".//joint"):
                sources.add(name)
                leaves.append(name)
                channel_sources.add(name)
        # The live skill resolver deliberately leaves a two-leaf root
        # ambiguous. Its ordinal exists, but it has no channel skill alias.
        if len(leaves) == 1:
            channel_sources.add(root_name)
    states = {row["object_name"] for row in modifications.get("articulation_states", [])}
    if states - bodies.keys():
        raise ValueError("legacy_registry_missing_articulated_body")
    sources.update(states)
    registry = {f"obj_{index:06d}": name for index, name in enumerate(sorted(sources), 1)}
    alias_counts = Counter(door_alias(oid) for oid, name in registry.items() if name in channel_sources)
    expected = {name for name, count in alias_counts.items() if count == 1}
    pattern = re.compile(r"\[v3-interaction-routing\].*?registered_channel_alias_count=(\d+) collision_count=(\d+) aliases=([^\r\n]*)")
    matches = pattern.findall(Path(eval_log).read_text(encoding="utf-8", errors="replace"))
    if len(matches) != 1:
        raise ValueError("legacy_registry_missing_or_ambiguous_routing_log")
    count, collisions, aliases = matches[0]
    recorded = {x.strip() for x in aliases.split(",") if x.strip()}
    if recorded != expected or int(count) != len(expected) or int(collisions) != sum(v > 1 for v in alias_counts.values()):
        raise ValueError("legacy_registry_routing_alias_mismatch")
    return registry


def recorded_candidate(manifest, opaque_id):
    """Return published geometry only for the verifier's accepted opaque ID."""
    if not manifest.is_file():
        return None
    opener = gzip.open if manifest.suffix == ".gz" else open
    observations = []
    with opener(manifest, "rt", encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            for obs in (row.get("gt_observations") or {}).get("observations", []):
                if str(obs.get("id") or obs.get("instance_id")) == opaque_id:
                    center = (obs.get("box_3d") or {}).get("center")
                    if center and len(center) == 3:
                        observations.append({"center": center, "step": row.get("step_index"), "name": obs.get("name")})
    return observations


def episode_decision(
    episode, result, result_path, attempt, scenes, radius, benchmark_hash,
    contract_audit=None,
):
    decision = {"accepted": False, "reason": "no_verified_alternative_claim"}
    if not result.get("goal_definition_relaxed_success"):
        return decision
    manifest_path = attempt / "eval/run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("benchmark_sha256") != benchmark_hash:
        return dict(decision, reason="benchmark_hash_mismatch")
    if manifest.get("protocol_version") != "interactive_nav_v3_benchmark_eval_v17":
        return dict(decision, reason="unsupported_legacy_registry_protocol")
    prefix = f"{episode['data_split']}_{episode['house_index']}"
    metadata_path, xml_path = scenes / f"{prefix}_metadata.json", scenes / f"{prefix}.xml"
    if not metadata_path.is_file() or not xml_path.is_file():
        return dict(decision, reason="missing_scene_metadata_or_xml")
    metadata = read_json(metadata_path)
    try:
        registry = reconstruct_v17_registry(episode, metadata, xml_path, attempt / "eval.log")
    except (ValueError, OSError, ET.ParseError) as exc:
        return dict(decision, reason=str(exc))
    candidate_id = result.get("goal_definition_relaxed_instance_id")
    candidate_name = registry.get(candidate_id)
    if not candidate_name:
        return dict(decision, reason="candidate_not_in_reconstructed_registry")
    target = episode["interactive_nav"]["target"]
    positions = dict(episode.get("scene_modifications", {}).get("object_poses", {}))
    sidecar = read_json(result_path.with_name("episode_visualization.json"))
    if sidecar.get("case_id") != result.get("case_id"):
        return dict(decision, reason="private_sidecar_case_mismatch")
    target_name = target["selected_instance"]
    runtime_target_xy = sidecar.get("target", {}).get("candidate_xy", {}).get(target_name)
    if runtime_target_xy and target_name in positions:
        positions[target_name] = list(runtime_target_xy) + list(positions[target_name][2:])
    public_manifest = attempt / "sim_step_frames/manifest.jsonl"
    if not public_manifest.is_file() and public_manifest.with_suffix(".jsonl.gz").is_file():
        public_manifest = public_manifest.with_suffix(".jsonl.gz")
    observations = recorded_candidate(public_manifest, candidate_id)
    identity_evidence = {
        "method": "v17_sorted_registry_xml_and_recorded_routing_crosscheck",
        "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
        "xml_path": str(xml_path), "xml_sha256": sha256(xml_path),
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "candidate_instance_id": candidate_id, "candidate_name": candidate_name,
        "public_frame_geometry_available": observations is not None,
    }
    # Use frozen object poses for equivalence, not an arbitrary final frame.
    # When recordings exist, cross-check the ordinal using the same public ID.
    if observations is not None:
        if not observations or candidate_name not in positions:
            return dict(decision, reason="candidate_geometry_not_recorded", identity_evidence=identity_evidence)
        discrepancies = [math.dist(obs["center"][:2], positions[candidate_name][:2]) for obs in observations]
        identity_evidence["minimum_recorded_xy_discrepancy_m"] = min(discrepancies)
        identity_evidence["last_recorded_center"] = observations[-1]["center"]
        if min(discrepancies) > 0.15:
            return dict(decision, reason="candidate_geometry_disagrees_with_registry", identity_evidence=identity_evidence)
    interaction_requirements_equal = False
    contract_audit_evidence = None
    if isinstance(contract_audit, dict):
        plan_id = str(contract_audit.get("plan_id") or "")
        audited_ids = [str(value) for value in contract_audit.get("required_interaction_ids", [])]
        plan = next(
            (
                row for row in episode["interactive_nav"].get("oracle_plans", [])
                if str(row.get("plan_id") or "") == plan_id
            ),
            None,
        )
        recorded_ids = [] if plan is None else [
            str(value) for value in plan.get("required_interaction_ids", [])
        ]
        interaction_requirements_equal = bool(
            contract_audit.get("reviewed") is True
            and str(contract_audit.get("candidate_name") or "") == candidate_name
            and plan is not None
            and audited_ids == recorded_ids
        )
        contract_audit_evidence = {
            "reviewed": contract_audit.get("reviewed") is True,
            "candidate_name_matches": str(contract_audit.get("candidate_name") or "") == candidate_name,
            "plan_id": plan_id,
            "plan_found": plan is not None,
            "required_interaction_ids_match": audited_ids == recorded_ids,
            "evidence": contract_audit.get("evidence"),
            "accepted": interaction_requirements_equal,
        }
    decision = equivalent_target(
        target=target,
        candidate_name=candidate_name,
        objects=metadata["objects"],
        positions=positions,
        near_radius_m=radius,
        interaction_requirements_equal=interaction_requirements_equal,
    )
    decision["identity_evidence"] = identity_evidence
    if contract_audit_evidence is not None:
        decision["contract_audit_evidence"] = contract_audit_evidence
    return decision


def aggregate(rows):
    n = len(rows)
    def rate(key, *fallback_keys, fallback_requires_nav=False):
        values = []
        for row in rows:
            primary = row.get(key) if row.get(key) is not None else None
            value = primary
            if value is None:
                for fallback_key in fallback_keys:
                    if row.get(fallback_key) is not None:
                        value = row.get(fallback_key)
                        break
            if fallback_requires_nav and primary is None:
                # Pre-layer rows sometimes contain a stale ``success`` claim
                # alongside nav_success=false.  Preserve the historical field
                # but keep the derived Interactive layer conservative.
                if row.get("nav_success") is False:
                    value = False
            if value is not None:
                values.append(value)
        return sum(bool(value) for value in values) / len(values) if values else None
    def mean(key):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None
    return {
        "episodes": n, "nav_successes": sum(r.get("nav_success") is True for r in rows),
        "interaction_conditioned_successes": sum(r.get("success") is True for r in rows),
        "nav_sr": rate("nav_success"), "interaction_conditioned_sr": rate("success"),
        "exact_instance_sr": rate("exact_instance_success", "nav_success"),
        "category_goal_sr": rate("category_goal_success", "nav_success"),
        "interaction_contract_goal_sr": rate(
            "interaction_contract_goal_success", "nav_success"
        ),
        "interactive_episode_sr": rate(
            "interactive_episode_success",
            "interaction_conditioned_success",
            "success",
            fallback_requires_nav=True,
        ),
        "required_interaction_sr": rate("required_interaction_success"),
        "spl_original_reference": mean("spl"), "mean_total_cost": mean("episode_total_cost"),
        "mean_interaction_precision": mean("interaction_precision_episode"),
        "mean_steps": mean("step_count"),
    }


def run(evaluation, benchmark, scenes, output, radius, contract_audit_path=None):
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing rescore directory: {output}")
    source_summary = evaluation / "summary.json"
    source = read_json(source_summary)
    episodes = read_json(benchmark)
    if isinstance(episodes, dict):
        episodes = episodes["episodes"]
    benchmark_hash = sha256(benchmark)
    contract_audit = None
    if contract_audit_path is not None:
        contract_audit = read_json(contract_audit_path)
        if contract_audit.get("schema_version") != "interactive_nav_contract_audit_v1":
            raise ValueError("Unsupported identity-contract audit schema")
        if contract_audit.get("benchmark_sha256") != benchmark_hash:
            raise ValueError("Identity-contract audit benchmark hash mismatch")
        if not isinstance(contract_audit.get("episodes"), dict):
            raise ValueError("Identity-contract audit episodes must be an object")
    strict, rescored, audits = [], [], []
    seen = set()
    for summary_row in source["episodes"]:
        index = int(summary_row["episode_index"])
        if index in seen:
            raise ValueError(f"Duplicate episode index {index}")
        seen.add(index)
        result_path = Path(summary_row["episode_result_path"])
        attempt = Path(summary_row["attempt_dir"])
        if not result_path.resolve().is_relative_to(evaluation.resolve()):
            raise ValueError(f"Episode {index} result points outside selected evaluation")
        document = read_json(result_path)
        result = document["result"]
        if result["episode_index"] != index or result["case_id"] != episodes[index]["interactive_nav"]["case_id"]:
            raise ValueError(f"Episode {index} identity mismatch")
        decision = episode_decision(
            episodes[index], result, result_path, attempt, scenes, radius,
            benchmark_hash,
            None if contract_audit is None else contract_audit["episodes"].get(str(index)),
        )
        revised = rescore_result(result, decision)
        strict.append(result)
        rescored.append(revised)
        audits.append({"episode_index": index, "case_id": result["case_id"],
                       "source_result": str(result_path), "source_sha256": sha256(result_path),
                       "original_nav_success": result.get("nav_success"),
                       "original_interaction_conditioned_success": result.get("success"),
                       "exact_instance_success": revised.get("exact_instance_success"),
                       "category_goal_success": revised.get("category_goal_success"),
                       "interaction_contract_goal_success": revised.get("interaction_contract_goal_success"),
                       "interactive_episode_success": revised.get("interactive_episode_success"),
                       **revised["goal_equivalence"]})
    groups = {}
    for group in ("all", "channel", "container", "mixed"):
        def belongs(row):
            domains = set(row.get("domains") or [])
            return group == "all" or (group == "mixed" and len(domains) > 1) or domains == {group}
        old, new = [r for r in strict if belongs(r)], [r for r in rescored if belongs(r)]
        groups[group] = {"strict_all": aggregate(old), "revised_all": aggregate(new),
                         "strict_eligible": aggregate([r for r in old if r.get("scoring_eligible") is True]),
                         "revised_eligible": aggregate([r for r in new if r.get("scoring_eligible") is True])}
    report = {
        "schema_version": PROTOCOL, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(source_summary), "source_summary_sha256": sha256(source_summary),
        "benchmark": str(benchmark), "benchmark_sha256": benchmark_hash,
        "contract_audit": (
            None
            if contract_audit_path is None
            else {"path": str(contract_audit_path), "sha256": sha256(contract_audit_path)}
        ),
        "implementation_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), Path(__file__).parent / "evaluation/goal_equivalence.py")},
        "rules": {"same_container": "category; contract requires frozen or candidate-specific plan proof",
                  "near_planar_radius_m": radius,
                  "near_same_category": "category_only_without_explicit_contract",
                  "same_room": "same-category category-only when room and parent/container evidence are consistent",
                  "shared_door": "does_not_prove_full_interaction_contract",
                  "different_known_container": "not_equivalent_without_explicit_contract",
                  "strict_metrics": "preserved",
                  "spl_reference": "not_computed_without_candidate_specific_reference"},
        "aggregates": groups, "decision_counts": dict(Counter(r["reason"] for r in audits)),
        "promoted_episode_indices": [r["episode_index"] for r in audits if r["promoted"]],
        "interaction_contract_promoted_episode_indices": [
            r["episode_index"] for r in audits if r["interaction_contract_promoted"]
        ],
        "interactive_episode_promoted_episode_indices": [
            r["episode_index"] for r in audits if r["interactive_episode_promoted"]
        ],
        "episodes": audits,
    }
    lines = ["# 分层目标与交互契约重评分（不改写原始严格指标）", "",
             f"同容器同类目标可进入相同交互契约；同房间同类目标（含容器内目标与台面目标）可计 Category，{radius:.2f} m 近邻也可计 Category。",
             "共享房间或同一道门不自动证明 container/full contract；仍要求公开观测、到达距离及候选对应的完整交互计划。", "",
             "原 `nav_success`、`success`、SPL 与 Total Cost 保持不变。类别候选没有重算最短路径，因此不生成 Category-SPL。", "",
             "| 分组 | N | Exact SR | Category SR | Contract SR | Interactive SR |",
             "|---|---:|---:|---:|---:|---:|"]
    def number(value):
        return "N/A" if value is None else f"{value:.4f}"
    for name, group in groups.items():
        for scope in ("all", "eligible"):
            new = group[f"revised_{scope}"]
            fields = [number(new[key]) for key in (
                "exact_instance_sr", "category_goal_sr",
                "interaction_contract_goal_sr", "interactive_episode_sr",
            )]
            lines.append(f"| {name}/{scope} | {new['episodes']} | " + " | ".join(fields) + " |")
    lines.extend(["", "## 类别目标新增成功场景", "",
                  "| Episode | 理由 | Category | Contract | Interactive |",
                  "|---|---|---|---|---|"])
    for audit in audits:
        if audit["promoted"]:
            lines.append(
                f"| {audit['episode_index']} | {audit['reason']} | 成功 | "
                f"{'成功' if audit['interaction_contract_goal_success'] else '失败'} | "
                f"{'成功' if audit['interactive_episode_success'] else '失败'} |"
            )
    unresolved = [r for r in audits if r["public_claim_verified"] and not r["accepted"]]
    lines.extend(["", "未接受的替代目标：", ""])
    lines.extend(f"- {r['episode_index']}: `{r['reason']}`" for r in unresolved)
    lines.extend(["", "原始 summary/result 未修改；详细身份映射、输入 SHA256 与逐场理由见 report.json。", ""])
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "rescored_results.json").write_text(json.dumps(rescored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Directory containing val_N_metadata.json and val_N.xml")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites existing results")
    parser.add_argument("--near-radius-m", type=float, default=0.30)
    parser.add_argument(
        "--identity-contract-audit",
        type=Path,
        help=(
            "Optional reviewed candidate-to-oracle-plan audit for legacy benchmarks "
            "that predate target.identity_contract."
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.near_radius_m) or args.near_radius_m <= 0:
        parser.error("--near-radius-m must be finite and positive")
    report = run(
        args.evaluation_dir.resolve(), args.benchmark.resolve(), args.scene_dir.resolve(),
        args.output_dir.resolve(), args.near_radius_m,
        None if args.identity_contract_audit is None else args.identity_contract_audit.resolve(),
    )
    print(json.dumps({"report": str(args.output_dir / "report.md"),
                      "promoted": report["promoted_episode_indices"],
                      "decision_counts": report["decision_counts"],
                      "all": report["aggregates"]["all"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
