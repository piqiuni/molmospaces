"""Replay two configured models with a shared request budget and frozen prompt.

Credentials are read in-process; saved artifacts contain model names and hashes,
never the key or gateway URL. This measures offline decision agreement with an
explicit rubric, not closed-loop navigation success.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.remove(str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shlex
import threading
import time
from typing import Any
from urllib.parse import quote, urlsplit

from scripts.InteractiveNav.evaluation import m2_replay_eval as replay


def read_connection(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.removeprefix("export ").split("=", 1)
        tokens = shlex.split(raw_value.strip(), comments=True)
        values[key.strip()] = tokens[0] if tokens else ""
    for key in ("url", "key", "model1", "model2"):
        if not values.get(key):
            raise ValueError(f"missing {key} in credential configuration")
    return values


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def redact_sensitive(value: Any, sensitive_values: tuple[str, ...]) -> Any:
    """Sanitize transport responses before any artifact writer sees them."""
    if isinstance(value, str):
        for secret in sensitive_values:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {
            redact_sensitive(key, sensitive_values): redact_sensitive(child, sensitive_values)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(child, sensitive_values) for child in value]
    return value


def credential_redactions(connection: dict[str, str]) -> tuple[str, ...]:
    endpoint = urlsplit(connection["url"])
    values = {connection["key"], connection["url"], connection["url"].rstrip("/")}
    if endpoint.netloc:
        values.update((endpoint.netloc, f"{endpoint.scheme}://{endpoint.netloc}"))
    if endpoint.hostname:
        values.add(endpoint.hostname)
    values.update(quote(value, safe="") for value in tuple(values))
    return tuple(sorted((value for value in values if value), key=len, reverse=True))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_inputs(
    output: Path,
    cases: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    prompt: str,
) -> dict[str, dict[str, str]]:
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    replay._write_jsonl(inputs / "cases.jsonl", cases)
    replay._write_jsonl(inputs / "annotations.jsonl", annotations.values())
    (inputs / "prompt.txt").write_text(prompt, encoding="utf-8")
    return {
        name: {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
        for name, path in (
            ("cases", inputs / "cases.jsonl"),
            ("annotations", inputs / "annotations.jsonl"),
            ("prompt", inputs / "prompt.txt"),
        )
    }


def rolling_max(timestamps: list[float], window: float = 60.0) -> int:
    ordered = sorted(timestamps)
    left, maximum = 0, 0
    for right, timestamp in enumerate(ordered):
        while timestamp - ordered[left] >= window:
            left += 1
        maximum = max(maximum, right - left + 1)
    return maximum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="/home/ldl/.env")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-minute", type=int, default=28)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--reasoning-effort", default="off")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-key", choices=("model1", "model2"), action="append")
    args = parser.parse_args(argv)
    if not 2 <= args.concurrency < 10:
        parser.error("combined remote concurrency must be in [2, 9]")
    if not 1 <= args.requests_per_minute < 30:
        parser.error("combined remote requests per minute must be in [1, 29]")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    connection = read_connection(Path(args.env_file))
    sensitive_values = credential_redactions(connection)
    os.environ["M2_MIRROR_REMOTE_API_KEY"] = connection["key"]
    dataset_path = Path(args.dataset).expanduser().resolve()
    if dataset_path.is_dir():
        dataset_path = dataset_path / "cases.jsonl"
    annotation_path = Path(args.annotations).expanduser().resolve()
    prompt_path = Path(args.prompt_file).expanduser().resolve()
    cases = replay.load_cases(dataset_path)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error("no cases selected")
    annotations = replay.load_annotations(annotation_path)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        parser.error("prompt file must not be empty")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "request_dispatch.jsonl").exists():
        parser.error("output directory already contains requests; select a fresh directory")
    model_keys = list(dict.fromkeys(args.model_key or ["model1", "model2"]))
    models = [connection[key] for key in model_keys]
    limiter = replay.SlidingWindowRateLimiter(args.requests_per_minute)
    event_lock = threading.Lock()
    dispatch_times: list[float] = []
    active, peak_active = 0, 0
    started = time.time()
    frozen_inputs = freeze_inputs(output, cases, annotations, prompt)
    effective_requests = [
        {"case_id": case["case_id"], "request": {**case["request"], "instruction": prompt}}
        for case in cases
    ]
    repository = Path(__file__).resolve().parents[3]
    source_files = (
        Path(__file__).resolve(), Path(replay.__file__).resolve(),
        repository / "Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts/semantic_mllm_py_pkg/client.py",
        repository / "Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/model_policy.py",
    )
    manifest = {
        "schema_version": "m2_mirror_comparison_v1",
        "models": models,
        "model_keys": model_keys,
        "case_count_per_model": len(cases),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "annotations_sha256": _sha256(annotation_path),
        "case_ids_sha256": hashlib.sha256(json.dumps([c["case_id"] for c in cases]).encode()).hexdigest(),
        "effective_public_requests_sha256": hashlib.sha256(json.dumps(
            effective_requests, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "effective_public_requests_note": "Before model-client transport formatting; frozen inputs and client source hash recorded.",
        "frozen_inputs": frozen_inputs,
        "source_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (("dataset", dataset_path), ("annotations", annotation_path), ("prompt", prompt_path))
        },
        "code_sha256": {str(path.relative_to(repository)): _sha256(path) for path in source_files},
        "concurrency_combined": args.concurrency,
        "requests_per_minute_combined": args.requests_per_minute,
        "timeout_s": args.timeout_s,
        "max_retries": args.max_retries,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "protocol": "openai_chat",
        "temperature": 0.0,
        "started_at": started,
        "metric_scope": "offline_public_context_agent_rubric_not_navigation_success",
    }
    write_json(output / "manifest.json", manifest)

    def run_model(index: int) -> dict[str, Any]:
        nonlocal active, peak_active
        model = models[index]
        model_output = output / model_keys[index]
        model_output.mkdir(parents=True, exist_ok=True)
        namespace = argparse.Namespace(
            mode="http", endpoint=connection["url"], api_key_env="M2_MIRROR_REMOTE_API_KEY",
            model=model, protocol="openai_chat", command="", timeout_s=args.timeout_s,
            temperature=0.0, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
        )
        # The client's default logger runs before this wrapper can redact.
        # Disable it and write only sanitized metrics below.
        call = replay.build_mllm_request_function(namespace, model_output, metrics_path="")
        transport_lock = threading.Lock()
        completed = 0
        prediction_file = model_output / "predictions.inprogress.jsonl"

        def request(payload: dict[str, Any], case_id: str):
            nonlocal active, peak_active
            with event_lock:
                now = time.time()
                dispatch_times.append(now)
                active += 1
                peak_active = max(peak_active, active)
                with (output / "request_dispatch.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"model": model, "case_id": case_id, "time": now,
                                             "active": active}) + "\n")
            try:
                request_started = time.monotonic()
                try:
                    response, metrics = call(payload, case_id)
                except Exception as exc:
                    response = None
                    metrics = {"error": f"{type(exc).__name__}: {exc}",
                               "latency_s": time.monotonic() - request_started}
                response = redact_sensitive(response, sensitive_values)
                metrics = redact_sensitive(dict(metrics), sensitive_values)
                record = {
                    "timestamp": time.time(), "role": "subgoal_selection", "model": model,
                    "timeout_s": args.timeout_s, "max_output_tokens": args.max_tokens,
                    "protocol": namespace.protocol, "reasoning_effort": args.reasoning_effort,
                    "case_id": case_id, "dataset_role": "m2_offline_replay", **metrics,
                }
                with transport_lock, (model_output / "transport_metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                return response, metrics
            finally:
                with event_lock:
                    active -= 1

        def on_prediction(row: dict[str, Any]) -> None:
            nonlocal completed
            completed += 1
            with prediction_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 10 == 0 or completed == len(cases):
                print(json.dumps({"model": model, "completed": completed, "total": len(cases),
                                  "elapsed_s": round(time.time() - started, 1)}, ensure_ascii=False), flush=True)

        predictions = replay.replay_cases(
            cases, request_function=request, annotations=annotations, max_retries=args.max_retries,
            prompt_override=prompt,
            concurrency=args.concurrency // len(models) + (args.concurrency % len(models) if index == 0 else 0),
            rate_limiter=limiter, on_prediction=on_prediction,
        )
        with (model_output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = replay.summarize_scores(predictions)
        summary.update(model=model, prompt_sha256=manifest["prompt_sha256"])
        write_json(model_output / "summary.json", summary)
        return summary

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        summaries = list(executor.map(run_model, range(len(models))))
    audit = {"dispatch_count_including_retries": len(dispatch_times),
             "max_requests_in_rolling_60s": rolling_max(dispatch_times),
             "peak_inflight_requests": peak_active, "wall_time_s": time.time() - started}
    audit["limits_respected"] = audit["max_requests_in_rolling_60s"] < 30 and peak_active < 10
    write_json(output / "rate_audit.json", audit)
    write_json(output / "summary.json", {"models": summaries, "rate_audit": audit})
    print(json.dumps({"models": summaries, "rate_audit": audit}, ensure_ascii=False), flush=True)
    if not audit["limits_respected"]:
        return 4
    return 0 if all(row["valid_response_count"] == len(cases) for row in summaries) else 3


if __name__ == "__main__":
    raise SystemExit(main())
