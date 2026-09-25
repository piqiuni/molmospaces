"""Freeze and run paired semantic-prompt experiments on public M2 snapshots."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import statistics
import threading
import time
from types import SimpleNamespace

from scripts.InteractiveNav.evaluation import m2_context_ablation as context
from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


ARMS = ("b", "p1", "p2", "p3", "p4", "p5", "p6", "s1_stage", "s2_region", "s3_diverse", "s_all")
STAGE_NOTE = """Additional stage-evidence check: robot.stage_facts summarizes recorded events and the CURRENT stage declaration. Use it to distinguish an unfinished dependency from possible stale or repeated instructions. A prior success for the same candidate or object is not proof that the CURRENT event is completed when event identity is unknown. STARTED is not failure or completion. Do not invent a completed stage, a fresh interaction, or a missing event ID; resolve only what the supplied evidence establishes."""
REGION_NOTE = """Additional regional-memory check: robot.region_history_facts groups recorded decisions by their explicit historical region identity. Selection counts are not physical visit counts. Use exact available region associations and recorded outcomes to avoid unchanged failed/no-gain repeats; geometric proximity alone does not establish the same region. Different room visits, unknown frame alignment, and missing outcomes must stay uncertain. A changed state, new information, or a different supplied approach may justify revisiting. Do not treat an entire previously entered room as exhausted."""
DIVERSITY_NOTE = """After choosing the best first action by the procedure above, select up to two genuinely useful fallbacks. Prefer alternatives that do not all depend on the same failed approach, blocked passage, or unchanged search region when comparably relevant alternatives are offered. Different IDs alone do not prove independent alternatives. Do not promote an unrelated or impossible action merely for diversity, and do not change the first action solely to diversify the list. Return fewer IDs when useful alternatives are absent; never pad or repeat an ID."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instruction_for(arm: str, prompts: dict[str, str]) -> str:
    if arm in prompts:
        return prompts[arm]
    additions = []
    if arm in {"s1_stage", "s_all"}:
        additions.append(STAGE_NOTE)
    if arm in {"s2_region", "s_all"}:
        additions.append(REGION_NOTE)
    if arm in {"s3_diverse", "s_all"}:
        additions.append(DIVERSITY_NOTE)
    if arm not in ARMS or not additions:
        raise ValueError(f"Unknown arm: {arm}")
    return prompts["p6"] + "\n\n" + "\n\n".join(additions)


def variant_request(request: dict, arm: str, prompts: dict[str, str], raw_record: dict | None = None) -> dict:
    from scripts.InteractiveNav.evaluation.m2_semantic_suggestions import enrich_request
    result = deepcopy(request)
    result["instruction"] = instruction_for(arm, prompts)
    if arm in {"s1_stage", "s2_region", "s_all"}:
        result = enrich_request(result, raw_record=raw_record,
                                stage=arm in {"s1_stage", "s_all"},
                                regions=arm in {"s2_region", "s_all"})
    if result["candidates"] != request["candidates"]:
        raise ValueError("A prompt experiment must not change candidate membership, facts or order")
    replay.audit_public_request(result)
    return result


def prepare(args: argparse.Namespace) -> dict:
    inputs, checks_dir, prompt_dir = map(Path, (args.inputs, args.checks, args.prompt_dir))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    historical = replay.load_cases(inputs / "cases.jsonl")
    if any(case.get("arm") != "full_context" for case in historical):
        raise ValueError("Expected the frozen full_context-only export")
    synthetic = replay.load_cases(checks_dir / "synthetic_cases.jsonl")
    labels_path = inputs / "base_annotations.jsonl"
    labels = replay.load_annotations(labels_path)
    records = {record["case_id"]: record for record in replay._jsonl_rows(Path(args.records))}
    prompt_manifest = json.loads((prompt_dir / "manifest.json").read_text())
    prompts = {"b": historical[0]["request"]["instruction"]}
    for arm in ARMS[1:7]:
        prompt_file = prompt_dir / prompt_manifest["prompts"][arm]["file"]
        if sha(prompt_file) != prompt_manifest["prompts"][arm]["file_sha256"]:
            raise ValueError(f"Prompt manifest hash mismatch: {arm}")
        prompts[arm] = prompt_file.read_text().strip()
        if not prompts[arm]:
            raise ValueError(f"Empty prompt: {arm}")
    if any(case["request"]["instruction"] != prompts["b"] for case in historical):
        raise ValueError("Historical baseline instructions differ")
    if set(labels) != {case["base_case_id"] for case in historical}:
        raise ValueError("Frozen historical labels do not match the source case set")
    if any(case["base_case_id"] not in records for case in historical):
        raise ValueError("Missing timestamp-bounded public source record")
    checks_manifest = json.loads((checks_dir / "manifest.json").read_text())
    if sha(inputs / "cases.jsonl") != checks_manifest["input_sha256"]:
        raise ValueError("Secondary checks were frozen for different inputs")
    for filename, expected in checks_manifest["file_sha256"].items():
        if sha(checks_dir / filename) != expected:
            raise ValueError(f"Frozen secondary check hash mismatch: {filename}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cases, budgets, token_rows = [], [], []
    for dataset, source_cases in (("historical", historical), ("synthetic", synthetic)):
        for source in source_cases:
            base = source.get("base_case_id") or source["case_id"]
            raw_record = records.get(base) if dataset == "historical" else source.get("public_evidence")
            for arm in ARMS:
                request = variant_request(source["request"], arm, prompts, raw_record)
                messages = context.wire_messages(request)
                token_count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                if token_count + args.max_tokens > args.max_model_len:
                    budgets.append({"source_case_id": source["case_id"], "arm": arm, "input_tokens": token_count})
                token_rows.append({"dataset": dataset, "arm": arm, "source_case_id": source["case_id"], "input_tokens": token_count})
                for repeat in range(1, args.repeats + 1):
                    cases.append({
                        "schema_version": replay.CASE_SCHEMA_VERSION,
                        "case_id": f"{arm}::r{repeat}::{dataset}::{base}",
                        "base_case_id": base, "source_case_id": source["case_id"],
                        "dataset": dataset, "arm": arm, "repeat": repeat,
                        "source": deepcopy(source.get("source") or {}),
                        "request": request, "request_sha256": context.digest(request),
                        "input_tokens_estimate": token_count,
                    })
    if budgets:
        raise ValueError(f"Token budget overflow; no inputs written or silently excluded: {budgets[:8]}")
    random.Random(args.seed).shuffle(cases)
    output.mkdir(parents=True, exist_ok=False)
    replay._write_jsonl(output / "cases.jsonl", cases)
    replay._write_jsonl(output / "token_budgets.jsonl", token_rows)
    replay._write_jsonl(output / "base_annotations.jsonl", labels.values())
    for name in ("checks.jsonl", "synthetic_checks.jsonl", "manifest.json", "rubric.md"):
        content = (checks_dir / name).read_bytes()
        destination = "checks_manifest.json" if name == "manifest.json" else name
        (output / destination).write_bytes(content)
    context.write_json(output / "prompts.json", {arm: instruction_for(arm, prompts) for arm in ARMS})
    repository = Path(__file__).resolve().parents[3]
    source_paths = [Path(__file__), Path(replay.__file__), Path(context.__file__),
                    Path(__file__).with_name("m2_semantic_suggestions.py"),
                    Path(__file__).with_name("m2_semantic_checks.py"),
                    repository / "Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts/semantic_mllm_py_pkg/client.py",
                    repository / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/model_policy.py"]
    manifest = {
        "schema_version": "m2_semantic_experiment_v1", "arms": list(ARMS), "repeats": args.repeats,
        "historical_case_count": len(historical), "synthetic_case_count": len(synthetic),
        "planned_requests": len(cases), "shuffle_seed": args.seed,
        "max_model_len": args.max_model_len, "max_output_tokens": args.max_tokens,
        "source_inputs": str(inputs.resolve()), "source_records": str(Path(args.records).resolve()),
        "source_labels_sha256": sha(labels_path), "source_cases_sha256": sha(inputs / "cases.jsonl"),
        "records_sha256": sha(Path(args.records)), "checks_manifest": json.loads((checks_dir / "manifest.json").read_text()),
        "prompt_manifest": prompt_manifest,
        "prompt_sha256": {arm: hashlib.sha256(instruction_for(arm, prompts).encode()).hexdigest() for arm in ARMS},
        "cases_sha256": context.digest(cases), "cases_file_sha256": sha(output / "cases.jsonl"),
        "generated_artifact_sha256": {name: sha(output / name) for name in (
            "cases.jsonl", "token_budgets.jsonl", "base_annotations.jsonl", "checks.jsonl",
            "synthetic_checks.jsonl", "checks_manifest.json", "rubric.md", "prompts.json")},
        "source_sha256": {str(path.relative_to(repository)): sha(path) for path in source_paths},
        "input_tokens": {arm: {"mean": statistics.fmean(row["input_tokens"] for row in token_rows if row["arm"] == arm),
                                "max": max(row["input_tokens"] for row in token_rows if row["arm"] == arm)} for arm in ARMS},
        "limits": ["Historical holdout has been inspected and is a regression split, not fresh blind evaluation",
                   "Synthetic strategy probes and historical scoring have separate denominators",
                   "Suggestion arms change evidence presentation as well as instruction; not pure prompt ablations"],
    }
    context.write_json(output / "manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict:
    inputs, output = Path(args.inputs), Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a fresh run directory; results must not be overwritten")
    cases = replay.load_cases(inputs / "cases.jsonl")
    frozen = json.loads((inputs / "manifest.json").read_text())
    if sha(inputs / "cases.jsonl") != frozen["cases_file_sha256"] or context.digest(cases) != frozen["cases_sha256"]:
        raise ValueError("Frozen input hash mismatch")
    for name, expected in frozen["generated_artifact_sha256"].items():
        if sha(inputs / name) != expected:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    if args.max_tokens != frozen["max_output_tokens"]:
        raise ValueError("Output budget must match the frozen preparation")
    labels = replay.load_annotations(inputs / "base_annotations.jsonl")
    annotations = {case["case_id"]: labels[case["base_case_id"]] for case in cases if case["dataset"] == "historical"}
    output.mkdir(parents=True, exist_ok=False)
    namespace = SimpleNamespace(mode="http", endpoint=args.endpoint, api_key_env="", model=args.model,
                                protocol="openai_chat", command="", timeout_s=args.timeout_s,
                                temperature=0.0, max_tokens=args.max_tokens, reasoning_effort="off")
    transport = replay.build_mllm_request_function(namespace, output)
    by_id = {case["case_id"]: case for case in cases}
    lock = threading.Lock()
    active = peak = completed = 0
    began = time.time()
    error_counts: Counter = Counter()
    manifest = {"model": args.model, "concurrency": args.concurrency, "temperature": 0.0,
                "reasoning_effort": "off", "timeout_s": args.timeout_s, "max_retries": args.max_retries,
                "max_tokens": args.max_tokens, "planned_requests": len(cases), "started_at": began,
                "inputs_manifest_sha256": sha(inputs / "manifest.json"), "cases_sha256": frozen["cases_sha256"],
                "source_sha256": {str(path): sha(path) for path in (Path(__file__), Path(replay.__file__))}}
    context.write_json(output / "manifest.json", manifest)

    def request(payload, case_id):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            with (output / "request_dispatch.jsonl").open("a") as stream:
                stream.write(json.dumps({"case_id": case_id, "time": time.time(), "active": active}) + "\n")
        try:
            return transport(payload, case_id)
        finally:
            with lock:
                active -= 1

    def on_prediction(row):
        nonlocal completed
        case = by_id[row["case_id"]]
        row.update({key: case[key] for key in ("base_case_id", "source_case_id", "arm", "repeat", "dataset")})
        with (output / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed += 1
        if row.get("transport_error"):
            error_counts["transport"] += 1
        elif not row.get("valid"):
            error_counts["schema"] += 1
        if completed % 64 == 0 or completed == len(cases):
            print(json.dumps({"completed": completed, "total": len(cases), "elapsed_s": round(time.time()-began, 1),
                              "active": active, "peak": peak, "errors": dict(error_counts)}), flush=True)

    replay.replay_cases(cases, request_function=request, annotations=annotations,
                        max_retries=args.max_retries, concurrency=args.concurrency, on_prediction=on_prediction)
    manifest.update(completed_at=time.time(), completed_requests=completed, peak_active=peak, errors=dict(error_counts))
    context.write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("inputs", "records", "checks", "prompt-dir", "tokenizer", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=1536)
    r = commands.add_parser("run")
    r.add_argument("--inputs", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    r.add_argument("--model", default="qwen3.6-35b-a3b-fp8")
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--timeout-s", type=float, default=120.0)
    r.add_argument("--max-retries", type=int, default=1)
    r.add_argument("--max-tokens", type=int, default=1536)
    args = parser.parse_args()
    print(json.dumps(prepare(args) if args.command == "prepare" else run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
