#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

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
        "Return JSON fields: object_id, interactable, interaction_class, coarse_state, "
        "interaction_parts (list of part_id/type/state/handle_visible/confidence), "
        "affordances, expected_effect, confidence, evidence_frame_ids."
    ),
    "subgoal_selection": (
        "Return JSON fields: candidate_id, scores, reason_tags. candidate_id must be one "
        "of the provided candidates."
    ),
    "skill_planning": (
        "Return JSON fields: object_id, subactions, max_retries. Each subaction contains "
        "skill, part_id, desired_state, and view_profile. Prefer opening a closed part before "
        "inspect_contents."
    ),
    "visual_verification": (
        "Return JSON fields: success, confidence, observed_states, new_contents_visible, "
        "retry_action, reason."
    ),
}


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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate configured MLLM models on the role question bank.")
    parser.add_argument("--question-bank", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default="")
    args = parser.parse_args()
    load_env_file(args.env_file or None)
    bank = json.loads(Path(args.question_bank).read_text(encoding="utf-8"))
    report = {"models": {}, "created_at": time.time()}
    for model in args.models:
        client = MLLMClient(client_config_from_env(model=model))
        rows = []
        for case in bank.get("cases") or []:
            response = client.request_json(
                role=str(case["role"]),
                instruction=(
                    str(case.get("instruction") or "")
                    + "\n"
                    + ROLE_GUIDANCE[str(case["role"])]
                    + " Return only JSON without Markdown."
                ),
                context=dict(case.get("context") or {}),
                images=list(case.get("image_paths") or []),
            )
            correct = False
            detail = {}
            if response.payload is not None and not response.error:
                try:
                    correct, detail = score_case(case, response.payload)
                except (ValueError, TypeError, KeyError) as exc:
                    detail = {"validation_error": str(exc)}
            rows.append({"case_id": case["id"], "role": case["role"], "correct": correct, "metrics": response.metrics(), "detail": detail, "error": response.error})
        per_role = {}
        for role in sorted({str(row["role"]) for row in rows}):
            per_role[role] = aggregate_rows(
                [row for row in rows if row["role"] == role]
            )
        report["models"][model] = {
            **aggregate_rows(rows),
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
            ],
        )
        writer.writeheader()
        for model, model_result in report["models"].items():
            for role, role_result in model_result["per_role"].items():
                writer.writerow({"model": model, "role": role, **role_result})
    print(output)


if __name__ == "__main__":
    main()
