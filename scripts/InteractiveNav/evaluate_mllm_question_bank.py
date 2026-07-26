#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import statistics
import time

import cv2

from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env, load_env_file
from semantic_mllm_py_pkg.schemas import (
    validate_attribute_patch,
    validate_skill_plan,
    validate_subgoal_selection,
    validate_visual_verification,
)


DEFAULT_MODELS = [
    "gpt-5.3-codex-spark",
    "qwen3.6-flash",
    "qwen3.5-35b-a3b",
    "deepseek-v4-flash",
]

ROLE_GUIDANCE = {
    "attribute_inference": (
        "Return compact JSON only. No explanation or reasoning. Fields: object_id, "
        "interactable, interaction_class, coarse_state, interaction_parts, confidence. "
        "interaction_class must be container, portal, none, or unknown. interaction_parts "
        "must be a JSON list of objects with part_id, type, state, and confidence. "
        "Use only visible evidence."
    ),
    "subgoal_selection": (
        "Return compact JSON only: {candidate_id: string}. Choose one provided candidate_id."
    ),
    "skill_planning": (
        "Return compact JSON only. Fields: object_id, subactions, max_retries. "
        "Each subaction has skill, part_id, desired_state. No explanation."
    ),
    "visual_verification": (
        "Inspect only the current post-interaction object crop. Return compact JSON only: "
        "{success: boolean, confidence: number, reason: string}. No explanation outside JSON."
    ),
}

COMPACT_ROLE_MAX_TOKENS = {
    "attribute_inference": 192,
    "subgoal_selection": 128,
    "skill_planning": 160,
    "visual_verification": 128,
}


def prepare_case_images(
    case: dict,
    crop_margin_ratio: float,
    crop_max_side_px: int,
) -> tuple[list[str], list[dict]]:
    paths = list(case.get("image_paths") or [])
    boxes = list(case.get("image_bboxes") or [])
    if str(case.get("role") or "") == "visual_verification":
        paths = paths[-1:]
        boxes = boxes[-1:] if boxes else []
    prepared = []
    metadata = []
    for index, image_path in enumerate(paths):
        path = Path(image_path).expanduser()
        box = boxes[index] if index < len(boxes) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            prepared.append(str(path))
            metadata.append(
                {
                    "path": str(path),
                    "cropped": False,
                    "source_bytes": path.stat().st_size if path.is_file() else 0,
                }
            )
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            prepared.append(str(path))
            metadata.append({"path": str(path), "cropped": False, "error": "read_failed"})
            continue
        height, width = image.shape[:2]
        raw_x0, raw_y0, raw_x1, raw_y1 = [int(round(float(value))) for value in box]
        left, right = sorted((raw_x0, raw_x1))
        top, bottom = sorted((raw_y0, raw_y1))
        margin_x = int(round((right - left) * max(0.0, crop_margin_ratio)))
        margin_y = int(round((bottom - top) * max(0.0, crop_margin_ratio)))
        left = max(0, min(width - 1, left - margin_x))
        right = min(width, max(left + 1, right + margin_x))
        top = max(0, min(height - 1, top - margin_y))
        bottom = min(height, max(top + 1, bottom + margin_y))
        crop = image[top:bottom, left:right]
        crop_height, crop_width = crop.shape[:2]
        scale = min(1.0, float(crop_max_side_px) / max(crop_width, crop_height))
        if scale < 1.0:
            crop = cv2.resize(
                crop,
                (
                    max(1, int(round(crop_width * scale))),
                    max(1, int(round(crop_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            prepared.append(str(path))
            metadata.append({"path": str(path), "cropped": False, "error": "encode_failed"})
            continue
        encoded_bytes = encoded.tobytes()
        prepared.append(
            "data:image/jpeg;base64," + base64.b64encode(encoded_bytes).decode("ascii")
        )
        metadata.append(
            {
                "path": str(path),
                "cropped": True,
                "source_size": [width, height],
                "bbox": [left, top, right, bottom],
                "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                "source_bytes": path.stat().st_size if path.is_file() else 0,
                "encoded_bytes": len(encoded_bytes),
            }
        )
    return prepared, metadata


def score_case(case: dict, payload: dict) -> tuple[bool, dict]:
    expected = case.get("expected") or {}
    role = case.get("role")
    if role == "attribute_inference":
        result = validate_attribute_patch(payload)
        correct = result.get("interactable") == expected.get("interactable")
        correct = correct and result.get("interaction_class") == expected.get("interaction_class")
        if expected.get("coarse_state"):
            correct = correct and result.get("coarse_state") == expected["coarse_state"]
        if expected.get("min_part_count"):
            correct = correct and len(result.get("interaction_parts") or []) >= int(expected["min_part_count"])
        detail = {"confidence": result.get("confidence"), "part_count": len(result.get("interaction_parts") or [])}
    elif role == "subgoal_selection":
        result = validate_subgoal_selection(payload, {str(item.get("candidate_id")) for item in case["context"].get("candidates", [])})
        correct = result.get("candidate_id") == expected.get("candidate_id")
        detail = {"selected": result.get("candidate_id"), "reason_tags": result.get("reason_tags", [])}
    elif role == "skill_planning":
        result = validate_skill_plan(payload, str(case["context"].get("object_id")))
        actions = result.get("subactions") or []
        correct = any(action.get("skill") == expected.get("required_skill") and action.get("part_id") == expected.get("required_part_id") for action in actions)
        detail = {"subactions": actions}
    elif role == "visual_verification":
        result = validate_visual_verification(payload)
        correct = result.get("success") == expected.get("success")
        detail = {"confidence": result.get("confidence"), "reason": result.get("reason")}
    else:
        raise ValueError(f"unsupported benchmark role: {role}")
    return bool(correct), detail


def aggregate_rows(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row["error"]]
    latencies = [float(row["metrics"].get("latency_s", 0.0)) for row in rows]

    def percentile(values: list[float], percentile_value: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        position = (len(sorted_values) - 1) * max(0.0, min(1.0, percentile_value))
        lower = int(position)
        upper = min(lower + 1, len(sorted_values) - 1)
        return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
            position - lower
        )

    def mean_metric(name: str, selected: list[dict]) -> float:
        return statistics.mean(float(row["metrics"].get(name, 0.0)) for row in selected) if selected else 0.0

    attribute_rows = [row for row in rows if row.get("role") == "attribute_inference"]

    def attribute_field_accuracy(field: str) -> float:
        comparable = [
            row
            for row in attribute_rows
            if field in (row.get("expected") or {}) and field in (row.get("prediction") or {})
        ]
        if not comparable:
            return 0.0
        return sum(
            row["prediction"][field] == row["expected"][field] for row in comparable
        ) / len(comparable)

    return {
        "case_count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "valid_response_rate": len(valid) / max(1, len(rows)),
        "mean_latency_s": statistics.mean(
            row["metrics"]["latency_s"] for row in rows
        )
        if rows
        else 0.0,
        "mean_valid_latency_s": statistics.mean(
            row["metrics"]["latency_s"] for row in valid
        )
        if valid
        else 0.0,
        "mean_valid_tps": statistics.mean(row["metrics"]["tps"] for row in valid)
        if valid
        else 0.0,
        "min_latency_s": min(latencies) if latencies else 0.0,
        "max_latency_s": max(latencies) if latencies else 0.0,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "mean_prompt_tokens": mean_metric("prompt_tokens", rows),
        "mean_completion_tokens": mean_metric("completion_tokens", rows),
        "mean_reasoning_tokens": mean_metric("reasoning_tokens", rows),
        "mean_visible_output_tokens": mean_metric("visible_output_tokens", rows),
        "attribute_interactable_accuracy": attribute_field_accuracy("interactable"),
        "attribute_class_accuracy": attribute_field_accuracy("interaction_class"),
        "attribute_state_accuracy": attribute_field_accuracy("coarse_state"),
    }


def evaluate_case(
    *,
    model: str,
    case: dict,
    role_max_tokens: dict[str, int],
    timeout_s: float,
    reasoning_effort: str,
    image_detail: str,
    crop_margin_ratio: float,
    crop_max_side_px: int,
) -> dict:
    """Run one isolated model request so concurrent requests never share a client."""
    started_at = time.perf_counter()
    role = str(case["role"])
    client_config = client_config_from_env(model=model)
    if timeout_s > 0.0:
        client_config.timeout_s = timeout_s
    client_config.reasoning_effort = reasoning_effort
    client_config.image_detail = image_detail
    client_config.max_tokens = role_max_tokens[role]
    client = MLLMClient(client_config)
    case_images: list[str] = []
    image_metadata: list[dict] = []
    correct = False
    detail: dict = {}
    prediction: dict = {}
    response = None
    try:
        case_images, image_metadata = prepare_case_images(
            case,
            crop_margin_ratio=crop_margin_ratio,
            crop_max_side_px=crop_max_side_px,
        )
        response = client.request_json(
            role=role,
            instruction=(
                str(case.get("instruction") or "")
                + "\n"
                + ROLE_GUIDANCE[role]
                + " Return only compact JSON without Markdown or reasoning."
            ),
            context=dict(case.get("context") or {}),
            images=case_images,
        )
        if response.payload is not None and not response.error:
            try:
                correct, detail = score_case(case, response.payload)
                if role == "attribute_inference":
                    normalized = validate_attribute_patch(response.payload)
                    prediction = {
                        key: normalized.get(key)
                        for key in ("interactable", "interaction_class", "coarse_state")
                    }
            except (ValueError, TypeError, KeyError) as exc:
                detail = {"validation_error": str(exc)}
    except Exception as exc:  # Keep a failed request as an evaluable row.
        detail = {"request_error": f"{type(exc).__name__}: {exc}"}

    elapsed_s = time.perf_counter() - started_at
    metrics = response.metrics() if response is not None else {
        "latency_s": elapsed_s,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "visible_output_tokens": 0,
        "total_tokens": 0,
        "tps": 0.0,
        "visible_output_tps": 0.0,
        "error": detail.get("request_error", "request_failed"),
    }
    return {
        "case_id": case["id"],
        "role": role,
        "correct": correct,
        "expected": dict(case.get("expected") or {}),
        "prediction": prediction,
        "metrics": metrics,
        "response": response.payload if response is not None else None,
        "raw_model_output_text": response.raw_text if response is not None else "",
        "raw_http_response": response.raw_http_response if response is not None else "",
        "usage": dict(response.usage) if response is not None else {},
        "detail": detail,
        "error": response.error if response is not None else detail["request_error"],
        "input_images": image_metadata,
        "request_elapsed_s": elapsed_s,
        "request_config": {
            "timeout_s": client_config.timeout_s,
            "max_output_tokens": client_config.max_tokens,
            "reasoning_effort": client_config.reasoning_effort,
            "wire_reasoning_effort": client._reasoning_effort_for_request(),
            "image_detail": client_config.image_detail,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate configured MLLM models on the role question bank.")
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--image-detail", choices=["low", "high", "auto", ""], default="low")
    parser.add_argument("--crop-margin-ratio", type=float, default=0.10)
    parser.add_argument("--crop-max-side-px", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent requests across every model and question (default: 1).",
    )
    args = parser.parse_args()
    load_env_file(args.env_file or None)
    bank = json.loads(Path(args.question_bank).read_text(encoding="utf-8"))
    role_max_tokens = dict(COMPACT_ROLE_MAX_TOKENS)
    if args.max_output_tokens > 0:
        role_max_tokens = {role: args.max_output_tokens for role in role_max_tokens}
    cases = list(bank.get("cases") or [])
    worker_count = max(1, min(int(args.workers), max(1, len(args.models) * len(cases))))
    report = {
        "models": {},
        "created_at": time.time(),
        "workers": worker_count,
        "parallel_scope": "all_model_case_requests",
    }
    rows_by_model: dict[str, dict[int, dict]] = {model: {} for model in args.models}
    model_started_at = {model: time.perf_counter() for model in args.models}
    model_finished_at = dict(model_started_at)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                evaluate_case,
                model=model,
                case=case,
                role_max_tokens=role_max_tokens,
                timeout_s=float(args.timeout_s),
                reasoning_effort=str(args.reasoning_effort or ""),
                image_detail=str(args.image_detail or ""),
                crop_margin_ratio=max(0.0, args.crop_margin_ratio),
                crop_max_side_px=max(64, args.crop_max_side_px),
            ): (model, case_index)
            for model in args.models
            for case_index, case in enumerate(cases)
        }
        for future in as_completed(futures):
            model, case_index = futures[future]
            rows_by_model[model][case_index] = future.result()
            model_finished_at[model] = time.perf_counter()

    for model in args.models:
        rows = [rows_by_model[model][case_index] for case_index in range(len(cases))]
        per_role = {}
        for role in sorted({str(row["role"]) for row in rows}):
            per_role[role] = aggregate_rows(
                [row for row in rows if row["role"] == role]
            )
        report["models"][model] = {
            **aggregate_rows(rows),
            "wall_time_s": model_finished_at[model] - model_started_at[model],
            "per_role": per_role,
            "rows": rows,
        }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "role",
                "case_count",
                "accuracy",
                "valid_response_rate",
                "mean_latency_s",
                "mean_valid_latency_s",
                "mean_valid_tps",
                "min_latency_s",
                "max_latency_s",
                "p50_latency_s",
                "p95_latency_s",
                "mean_prompt_tokens",
                "mean_completion_tokens",
                "mean_reasoning_tokens",
                "mean_visible_output_tokens",
                "attribute_interactable_accuracy",
                "attribute_class_accuracy",
                "attribute_state_accuracy",
                "wall_time_s",
            ],
        )
        writer.writeheader()
        for model, model_result in report["models"].items():
            for role, role_result in model_result["per_role"].items():
                writer.writerow({"model": model, "role": role, **role_result})
    print(output)


if __name__ == "__main__":
    main()
