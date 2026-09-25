"""Freeze and run paired Qwen context ablations without changing the prompt."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.remove(str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import random
import statistics
import threading
import time
from typing import Any

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


ARM_NAMES = (
    "full_context", "compact_control", "no_recent_decisions", "no_room_route",
    "no_geometry", "no_room_frontiers", "no_key_objects", "with_pre_scores",
    "legacy_candidate_pool",
)
GEOMETRY_KEYS = {
    "initial_xy", "current_xy", "entry_xy", "goal_xy", "xy", "center_xy",
    "centroid_xy", "aabb_center_xy", "aabb_size_xy", "aabb_size", "position_xy", "observed_xy",
}
FRONTIER_SUMMARY_KEYS = {
    "observed_frontier_length_m", "eligible_frontier_length_m",
    "eligible_frontier_count", "observed_frontier_count", "frontier_length_m",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_keys(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: remove_keys(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [remove_keys(v, keys) for v in value]
    return value


def context_variant(request: dict[str, Any], arm: str) -> dict[str, Any]:
    """Ablate explicit fields only; do not change candidate order or instruction."""
    result = deepcopy(request)
    if arm in {"no_recent_decisions", "compact_control"}:
        result["recent_decisions"] = []
    if arm in {"no_room_route", "compact_control"}:
        result["robot"].pop("room_visit_history", None)
    if arm in {"no_geometry", "compact_control"}:
        result = remove_keys(result, GEOMETRY_KEYS)
    if arm in {"no_room_frontiers", "compact_control"}:
        for room in result.get("graph", {}).get("rooms", []):
            for key in FRONTIER_SUMMARY_KEYS:
                room.pop(key, None)
        for candidate in result.get("candidates", []):
            candidate.pop("room_frontier_length_m", None)
    if arm == "no_key_objects":
        for room in result.get("graph", {}).get("rooms", []):
            room.pop("anchor_objects", None)
            room.pop("key_objects", None)
        for key in ("unassigned_anchor_objects", "remembered_objects_without_room"):
            result.get("graph", {}).pop(key, None)
    return result


def build_arm_requests(record: dict[str, Any], prompt: str) -> tuple[dict, dict]:
    replay._install_runtime_import_paths()
    from semantic_decision_py_pkg.behavior_candidates import (
        BehaviorCandidate, execution_candidates_with_reobserve_fallback,
    )
    from semantic_decision_py_pkg.candidate_curator import CandidateCurator, CandidateCuratorConfig
    from semantic_decision_py_pkg.model_policy import (
        ModelPolicyClient, ModelPolicyConfig, aggregate_room_frontier_lengths, _candidate_room_id,
    )

    graph = record["graph"]
    robot = deepcopy(record["robot_context"])
    target = record["target_context"]
    observation_step = int(robot.get("observation_step", record["source"].get("recorder_step")) or 0)
    for history in (robot.get("candidate_history") or {}).values():
        if "last_selected_step" in history:
            history["last_selected_steps_ago"] = max(0, observation_step - int(history["last_selected_step"]))
    selected_requests, diagnostics = {}, {}
    for pool_mode in ("all_actions_12_frontiers", "legacy"):
        raw = [BehaviorCandidate(**deepcopy(item)) for item in record["raw_candidates"]]
        curator = CandidateCurator(CandidateCuratorConfig(pool_mode=pool_mode))
        eligible, hard_rejections = curator.filter_candidates(
            raw, graph=graph, history_by_key=robot.get("candidate_history") or {},
            observation_step=observation_step,
        )
        eligible, deferred_reobserve_ids = execution_candidates_with_reobserve_fallback(eligible)
        curation = curator.curate(
            eligible, graph=graph, history_by_key=robot.get("candidate_history") or {},
            observation_step=observation_step,
            target_context=target, entered_room_ids=robot.get("entered_room_ids") or [],
        )
        if not curation.candidates:
            raise ValueError(f"{record['case_id']}: empty {pool_mode} pool")
        context = deepcopy(robot)
        context["room_frontier_lengths"] = aggregate_room_frontier_lengths(eligible, graph)
        counts = Counter(_candidate_room_id(c, graph)
                         for c in eligible if c.behavior_type == "EXPLORE")
        context["room_frontier_counts"] = {(key if key.startswith("room_") else "room_" + key): value
                                           for key, value in counts.items() if key}
        context["frontier_statistics_source"] = {
            "observed_scope": "recorded_public_raw_candidate_pool_lower_bound",
            "eligible_scope": "public_hard_validation_and_reobserve_fallback_before_M2_curation",
            "complete_map_frontier_inventory": False,
        }
        context.update(
            candidate_pre_scores=curation.quality_by_id,
            candidate_pre_score_terms=curation.quality_terms_by_id,
            candidate_decision_hints=curation.decision_hint_by_id,
        )
        client = ModelPolicyClient(ModelPolicyConfig(mode="disabled", include_pre_scores=False))
        request = client.build_request(curation.candidates, target, graph, context)
        request["instruction"] = prompt
        selected_requests[pool_mode] = request
        if pool_mode == "all_actions_12_frontiers":
            scored_client = ModelPolicyClient(ModelPolicyConfig(mode="disabled", include_pre_scores=True))
            scored = scored_client.build_request(curation.candidates, target, graph, context)
            scored["instruction"] = prompt
            selected_requests["with_pre_scores"] = scored
        diagnostics[pool_mode] = {
            "candidate_ids": [c.candidate_id for c in curation.candidates],
            "candidate_count_by_type": dict(Counter(c.behavior_type for c in curation.candidates)),
            "rejected": curation.rejected, "omitted": curation.omitted,
            "pre_curation_hard_rejections": hard_rejections,
            "deferred_reobserve_ids": deferred_reobserve_ids,
            "eligibility_limitations": "Unrecorded online mission/approach-exhaustion state cannot be exactly replayed.",
            "quality_by_id": curation.quality_by_id,
            "quality_terms_by_id": curation.quality_terms_by_id,
            "decision_hint_by_id": curation.decision_hint_by_id,
        }
    full = selected_requests["all_actions_12_frontiers"]
    arms = {name: context_variant(full, name) for name in ARM_NAMES}
    arms["legacy_candidate_pool"] = selected_requests["legacy"]
    arms["with_pre_scores"] = selected_requests["with_pre_scores"]
    for name, request in arms.items():
        replay.audit_public_request(request)
        assert request["instruction"] == prompt
        if name != "legacy_candidate_pool":
            assert [c["id"] for c in request["candidates"]] == [c["id"] for c in full["candidates"]]
    return arms, diagnostics


def wire_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    # Same text-only openai_chat serialization as the production MLLMClient.
    instruction = request["instruction"]
    if "/no_think" not in instruction:
        instruction = instruction.rstrip() + "\n/no_think"
    context = {key: request[key] for key in (
        "mission", "robot", "graph", "room_object_reasoning", "candidates", "recent_decisions",
    )}
    return [
        {"role": "system", "content": "Return only a valid JSON object."},
        {"role": "user", "content": [{"type": "text", "text": instruction + "\n" + json.dumps(context, ensure_ascii=False)}]},
    ]


def export_examples(inputs: Path) -> None:
    rows = [row for row in replay._jsonl_rows(inputs / "cases.jsonl") if row["arm"] == "full_context"]
    mean_tokens = statistics.fmean(row["input_tokens_estimate"] for row in rows)
    chosen = {
        "average_like": min(rows, key=lambda row: abs(row["input_tokens_estimate"] - mean_tokens)),
        "maximum": max(rows, key=lambda row: row["input_tokens_estimate"]),
    }
    for name, row in chosen.items():
        folder = inputs / "examples" / name
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "context.json", row["request"])
        messages = wire_messages(row["request"])
        write_json(folder / "messages.json", messages)
        write_json(folder / "metadata.json", {key: value for key, value in row.items() if key != "request"})
        text = f"# {name}\n\nCase: `{row['base_case_id']}`\n\nInput tokens (local tokenizer): {row['input_tokens_estimate']}\n\n"
        text += "## System\n\n```text\n" + messages[0]["content"] + "\n```\n\n"
        text += "## User — complete text, not abbreviated\n\n```text\n" + messages[1]["content"][0]["text"] + "\n```\n"
        (folder / "complete_input.md").write_text(text, encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "cases.jsonl").exists():
        raise ValueError("Use a fresh output directory; frozen cases already exist")
    records = list(replay._jsonl_rows(Path(args.records)))
    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    frozen, masters, diagnostics, excluded = [], [], [], []
    for record in records:
        try:
            arms, diagnostic = build_arm_requests(record, prompt)
            counts = {name: len(tokenizer.apply_chat_template(
                wire_messages(request), tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )) for name, request in arms.items()}
            if max(counts.values()) + args.max_tokens > args.max_model_len:
                raise ValueError(f"context_window_exceeded: {counts}")
        except (ValueError, TypeError, KeyError) as exc:
            excluded.append({"case_id": record["case_id"], "error": str(exc)})
            continue
        source = {**record["source"], "reconstruction": record["reconstruction"]}
        union = {c["id"]: c for c in arms["legacy_candidate_pool"]["candidates"]}
        union.update({c["id"]: c for c in arms["full_context"]["candidates"]})
        master_request = deepcopy(arms["full_context"])
        master_request["candidates"] = [union[k] for k in sorted(union)]
        masters.append({"case_id": record["case_id"], "source": source, "request": master_request})
        for name, request in arms.items():
            frozen.append({
                "schema_version": replay.CASE_SCHEMA_VERSION,
                "case_id": f"{name}::{record['case_id']}", "base_case_id": record["case_id"],
                "arm": name, "source": source, "request": request,
                "input_tokens_estimate": counts[name], "request_sha256": digest(request),
            })
        diagnostics.append({"case_id": record["case_id"], "pools": diagnostic})
    if not masters:
        raise ValueError(f"No admissible paired cases: {excluded[:3]}")
    random.Random(20260922).shuffle(frozen)
    replay._write_jsonl(output / "cases.jsonl", frozen)
    replay._write_jsonl(output / "label_inputs.jsonl", masters)
    replay._write_jsonl(output / "curation_diagnostics.jsonl", diagnostics)
    replay._write_jsonl(output / "excluded.jsonl", excluded)
    (output / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "m2_context_ablation_inputs_v1", "arms": list(ARM_NAMES),
        "source_records": len(records), "paired_cases": len(masters), "requests": len(frozen),
        "excluded_cases": len(excluded), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "cases_sha256": digest(frozen), "label_inputs_sha256": digest(masters),
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "input_tokens": {name: {"mean": statistics.fmean([r["input_tokens_estimate"] for r in frozen if r["arm"] == name]),
                                       "max": max(r["input_tokens_estimate"] for r in frozen if r["arm"] == name)} for name in ARM_NAMES},
        "control_note": "compact_control is a causal facts-only compact projection of the same expanded pool, not the historical request.",
        "score_scope": "raw Qwen ranking; production post-score guard not applied",
    }
    write_json(output / "manifest.json", manifest)
    export_examples(output)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    inputs, output = Path(args.inputs).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.jsonl").exists():
        raise ValueError("Use a fresh run directory")
    cases = list(replay._jsonl_rows(inputs / "cases.jsonl"))
    labels = replay.load_annotations(Path(args.annotations))
    if args.limit:
        selected_ids = sorted({case["base_case_id"] for case in cases})[:args.limit]
        cases = [case for case in cases if case["base_case_id"] in selected_ids]
    if args.arm:
        cases = [case for case in cases if case["arm"] in args.arm]
    annotations = {case["case_id"]: labels[case["base_case_id"]] for case in cases}
    # Fail before starting requests if labels for any selected case are missing.
    frozen_labels = [labels[key] for key in sorted({case["base_case_id"] for case in cases})]
    replay._write_jsonl(output / "frozen_labels.jsonl", frozen_labels)
    namespace = argparse.Namespace(
        mode="http", endpoint=args.endpoint, api_key_env="", model=args.model,
        protocol="openai_chat", command="", timeout_s=args.timeout_s,
        temperature=0.0, max_tokens=args.max_tokens, reasoning_effort="off",
    )
    transport = replay.build_mllm_request_function(namespace, output)
    lock = threading.Lock()
    active = peak_active = completed = 0
    started = time.time()
    manifest = {
        "model": args.model, "concurrency": args.concurrency, "planned_requests": len(cases),
        "started_at": started, "max_tokens": args.max_tokens, "timeout_s": args.timeout_s,
        "reasoning_effort": "off", "temperature": 0.0, "max_retries": args.max_retries,
        "cases_sha256": digest(cases), "labels_sha256": digest(frozen_labels),
        "prompt_sha256": hashlib.sha256(cases[0]["request"]["instruction"].encode()).hexdigest(),
    }
    repository = Path(__file__).resolve().parents[3]
    code_paths = [Path(__file__), Path(replay.__file__),
        repository / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/model_policy.py",
        repository / "Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts/semantic_mllm_py_pkg/client.py"]
    manifest["source_sha256"] = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    write_json(output / "manifest.json", manifest)
    by_id = {case["case_id"]: case for case in cases}

    def request(payload, case_id):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(active, peak_active)
            with (output / "request_dispatch.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"case_id": case_id, "time": time.time(), "active": active}) + "\n")
        try:
            return transport(payload, case_id)
        finally:
            with lock:
                active -= 1

    def on_prediction(row):
        nonlocal completed
        case = by_id[row["case_id"]]
        row.update(arm=case["arm"], base_case_id=case["base_case_id"])
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed += 1
        if completed % 32 == 0 or completed == len(cases):
            print(json.dumps({"completed": completed, "total": len(cases), "elapsed_s": round(time.time()-started, 1), "active": active, "peak": peak_active}), flush=True)

    predictions = replay.replay_cases(
        cases, request_function=request, annotations=annotations,
        max_retries=args.max_retries, concurrency=args.concurrency, on_prediction=on_prediction,
    )
    summary = {name: replay.summarize_scores([p for p in predictions if p["arm"] == name])
               for name in sorted({case["arm"] for case in cases})}
    write_json(output / "summary.json", summary)
    manifest.update(completed_at=time.time(), completed_requests=len(predictions), peak_active=peak_active)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def export_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Export one frozen arm for the existing rate-limited remote runner."""
    inputs = Path(args.inputs).resolve()
    labels_path = Path(args.annotations).resolve()
    cases = [case for case in replay.load_cases(inputs / "cases.jsonl")
             if case.get("arm") == args.arm]
    if not cases:
        raise ValueError("No cases matched the requested arm")
    labels = replay.load_annotations(labels_path)
    prompts = {case["request"]["instruction"] for case in cases}
    if len(prompts) != 1:
        raise ValueError("Selected cases do not share one frozen instruction")
    annotations = []
    for case in cases:
        if digest(case["request"]) != case["request_sha256"]:
            raise ValueError(f"Frozen request hash mismatch: {case['case_id']}")
        annotation = deepcopy(labels[case["base_case_id"]])
        annotation["base_case_id"] = case["base_case_id"]
        annotation["case_id"] = case["case_id"]
        annotations.append(annotation)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay._write_jsonl(output / "cases.jsonl", cases)
    replay._write_jsonl(output / "annotations.jsonl", annotations)
    replay._write_jsonl(output / "base_annotations.jsonl", [labels[c["base_case_id"]] for c in cases])
    prompt = next(iter(prompts))
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    manifest = {
        "arm": args.arm, "case_count": len(cases),
        "source_inputs": str(inputs), "source_annotations": str(labels_path),
        "cases_sha256": digest(cases),
        "source_annotations_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "annotation_change": "Namespace case_id and add base_case_id metadata; acceptable IDs, split and eligibility unchanged",
        "wire_messages_sha256": digest([{"case_id": c["case_id"], "messages": wire_messages(c["request"])} for c in cases]),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--records", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-model-len", type=int, default=10240)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--output-dir", required=True)
    r = commands.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--annotations", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    r.add_argument("--model", default="qwen3.6-35b-a3b-fp8")
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--timeout-s", type=float, default=120)
    r.add_argument("--max-tokens", type=int, default=1536)
    r.add_argument("--max-retries", type=int, default=1)
    r.add_argument("--limit", type=int)
    r.add_argument("--arm", action="append", choices=ARM_NAMES)
    e = commands.add_parser("examples")
    e.add_argument("--inputs", required=True)
    x = commands.add_parser("export-arm")
    x.add_argument("--inputs", required=True)
    x.add_argument("--annotations", required=True)
    x.add_argument("--arm", choices=ARM_NAMES, default="full_context")
    x.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "examples":
        export_examples(Path(args.inputs).resolve())
    elif args.command == "export-arm":
        print(json.dumps(export_arm(args), ensure_ascii=False), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()
