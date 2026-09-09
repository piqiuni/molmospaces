#!/usr/bin/env python3
"""Measure local OpenAI-compatible Qwen service concurrency without external I/O.

The benchmark reports end-to-end visible-token throughput rather than streamed
TTFT, because the navigation clients use non-streaming chat completions too.
It measures each endpoint alone and then the aggregate round-robin pool.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import mimetypes
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TEXT_PROMPT = (
    "You are a robot navigation policy component. Return exactly one compact JSON "
    "object with keys action and confidence. Choose action from navigate, scan, "
    "interact, or wait. Do not explain your answer."
)
VISION_PROMPT = (
    "Inspect this robot head-camera image. Return exactly one compact JSON object "
    "with keys visible_interactable, likely_orientation, and confidence. Do not "
    "explain your answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("text", "vision", "both"), default="both")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Requests per concurrent slot in a pooled measurement.",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--api-key", default="local")
    return parser.parse_args()


def encode_image(path: Path | None) -> str | None:
    if path is None:
        return None
    payload = path.expanduser().resolve().read_bytes()
    if not payload:
        raise ValueError(f"Image is empty: {path}")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def request_payload(model: str, max_tokens: int, mode: str, image_url: str | None) -> dict[str, Any]:
    """Build the same OpenAI-chat request shape used by the navigation nodes."""
    instruction = TEXT_PROMPT + "\n/no_think"
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    if mode == "vision":
        if not image_url:
            raise ValueError("--image is required for vision benchmarking")
        content = [
            {"type": "text", "text": VISION_PROMPT + "\n/no_think"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only a valid JSON object."},
            {"role": "user", "content": content},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "none",
        "enable_thinking": False,
        "stream": False,
    }


def visible_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def completion_tokens(response: dict[str, Any]) -> int:
    usage = response.get("usage") or {}
    value = usage.get("completion_tokens")
    if isinstance(value, int) and value >= 0:
        return value
    return len(visible_text(response).split())


def call_endpoint(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
    api_key: str,
    request_id: int,
) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        finished = time.perf_counter()
        tokens = completion_tokens(decoded)
        return {
            "request_id": request_id,
            "endpoint": endpoint,
            "ok": True,
            "started_s": started,
            "finished_s": finished,
            "latency_s": finished - started,
            "completion_tokens": tokens,
            "visible_chars": len(visible_text(decoded)),
            "error": "",
        }
    except HTTPError as exc:
        finished = time.perf_counter()
        detail = exc.read().decode("utf-8", errors="replace")[:1_000]
        return {
            "request_id": request_id,
            "endpoint": endpoint,
            "ok": False,
            "started_s": started,
            "finished_s": finished,
            "latency_s": finished - started,
            "completion_tokens": 0,
            "visible_chars": 0,
            "error": f"HTTPError: {exc}; body={detail}",
        }
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        finished = time.perf_counter()
        return {
            "request_id": request_id,
            "endpoint": endpoint,
            "ok": False,
            "started_s": started,
            "finished_s": finished,
            "latency_s": finished - started,
            "completion_tokens": 0,
            "visible_chars": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def summarize(label: str, mode: str, concurrency: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["ok"]]
    latencies = [float(record["latency_s"]) for record in successful]
    makespan_s = (
        max(float(record["finished_s"]) for record in records)
        - min(float(record["started_s"]) for record in records)
        if records
        else 0.0
    )
    tokens = sum(int(record["completion_tokens"]) for record in successful)
    return {
        "label": label,
        "mode": mode,
        "concurrency": int(concurrency),
        "request_count": len(records),
        "success_count": len(successful),
        "error_count": len(records) - len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "makespan_s": makespan_s,
        "requests_per_s": len(successful) / makespan_s if makespan_s > 0 else 0.0,
        "completion_tokens": tokens,
        "visible_output_tps": tokens / makespan_s if makespan_s > 0 else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "records": records,
    }


def run_measurement(
    *,
    label: str,
    endpoints: list[str],
    concurrency: int,
    request_count: int,
    mode: str,
    payload: dict[str, Any],
    timeout_s: float,
    api_key: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                call_endpoint,
                endpoints[index % len(endpoints)],
                payload,
                timeout_s,
                api_key,
                index,
            )
            for index in range(request_count)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: int(record["request_id"]))
    return summarize(label, mode, concurrency, records)


def main() -> int:
    args = parse_args()
    endpoints = [endpoint.rstrip("/") for endpoint in args.endpoints if endpoint.strip()]
    if not endpoints:
        raise ValueError("At least one endpoint is required")
    if any(value < 1 for value in args.concurrencies):
        raise ValueError("--concurrencies values must be positive")
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    modes = [args.mode] if args.mode != "both" else ["text", "vision"]
    image_url = encode_image(args.image)
    if "vision" in modes and not image_url:
        raise ValueError("--image is required when --mode vision or both")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for mode in modes:
        payload = request_payload(args.model, args.max_tokens, mode, image_url)
        # Exclude model loading / graph capture from measured runs.
        for endpoint in endpoints:
            warmup = call_endpoint(endpoint, payload, args.timeout_s, args.api_key, -1)
            if not warmup["ok"]:
                raise RuntimeError(f"Warmup failed for {endpoint}: {warmup['error']}")
        for endpoint_index, endpoint in enumerate(endpoints):
            results.append(
                run_measurement(
                    label=f"{mode}_single_gpu{endpoint_index}",
                    endpoints=[endpoint],
                    concurrency=1,
                    request_count=max(3, int(args.rounds)),
                    mode=mode,
                    payload=payload,
                    timeout_s=args.timeout_s,
                    api_key=args.api_key,
                )
            )
        for concurrency in args.concurrencies:
            results.append(
                run_measurement(
                    label=f"{mode}_pool_c{concurrency}",
                    endpoints=endpoints,
                    concurrency=concurrency,
                    request_count=max(len(endpoints), int(concurrency) * int(args.rounds)),
                    mode=mode,
                    payload=payload,
                    timeout_s=args.timeout_s,
                    api_key=args.api_key,
                )
            )

    payload = {
        "schema_version": "local_qwen_concurrency_v1",
        "created_unix_s": time.time(),
        "endpoints": endpoints,
        "model": args.model,
        "max_tokens": int(args.max_tokens),
        "rounds": int(args.rounds),
        "results": results,
    }
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "label",
        "mode",
        "concurrency",
        "request_count",
        "success_count",
        "error_count",
        "success_rate",
        "makespan_s",
        "requests_per_s",
        "completion_tokens",
        "visible_output_tps",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({key: value for key, value in payload.items() if key != "results"} | {"results": [{key: row[key] for key in fields} for row in results]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
