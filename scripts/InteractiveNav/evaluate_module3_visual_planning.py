#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
MLLM_SCRIPTS = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_mllm_py_pkg"
    / "scripts"
)
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))


from semantic_mllm_py_pkg.client import MLLMClient
from semantic_mllm_py_pkg.env import client_config_from_env, load_env_file
from semantic_mllm_py_pkg.interaction_prompt import (
    VISUAL_INTERACTION_PLANNING_INSTRUCTION,
    visual_interaction_planning_context,
)
from semantic_mllm_py_pkg.schemas import validate_visual_interaction_plan


def _image_data_url(image_path: Path, crop_box: list[int] | None) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if crop_box:
        x0, y0, x1, y1 = [int(value) for value in crop_box]
        height, width = image.shape[:2]
        left, right = sorted((max(0, x0), min(width, x1)))
        top, bottom = sorted((max(0, y0), min(height, y1)))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid crop for {image_path}: {crop_box}")
        image = image[top:bottom, left:right]
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"Cannot encode image: {image_path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _resolve_image(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else manifest_path.parent / path


def _evaluate_case(client: MLLMClient, manifest_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    expected_type = str(case.get("expected_target_type") or "unknown")
    response = client.request_json(
        role="skill_planning",
        instruction=VISUAL_INTERACTION_PLANNING_INSTRUCTION,
        context=visual_interaction_planning_context(
            object_id=str(case.get("object_id") or case.get("name") or "historical_target"),
            object_name=str(case.get("object_name") or case.get("name") or "historical_target"),
            expected_target_type=expected_type,
            requested_action=str(case.get("requested_action") or "open"),
        ),
        images=[
            _image_data_url(
                _resolve_image(manifest_path, str(case["image"])),
                list(case.get("crop") or []) or None,
            )
        ],
        timeout_s=float(case.get("timeout_s", client.config.timeout_s)),
        max_tokens=int(case.get("max_tokens", 256)),
    )
    result: dict[str, Any] = {
        "name": str(case.get("name") or "case"),
        "image": str(case["image"]),
        "crop": list(case.get("crop") or []),
        "expected_target_type": expected_type,
        "model_error": response.error,
        "raw_text": response.raw_text,
        "metrics": response.metrics(),
        "schema_valid": False,
        "checks": {},
    }
    if response.payload is None or response.error:
        return result
    try:
        plan = validate_visual_interaction_plan(
            response.payload,
            expected_target_type=expected_type,
            requested_action=str(case.get("requested_action") or "open"),
        )
    except ValueError as exc:
        result["validation_error"] = str(exc)
        return result
    result["schema_valid"] = True
    result["plan"] = plan
    checks = {
        "target_type_match": plan["target_type"] == expected_type,
        "centers_in_bounds": all(
            0.0 <= float(region["center"][0]) <= 1.0
            and 0.0 <= float(region["center"][1]) <= 1.0
            for region in plan["open_regions"]
        ),
        "centers_top_to_bottom": [region["center"][1] for region in plan["open_regions"]]
        == sorted(region["center"][1] for region in plan["open_regions"]),
    }
    if case.get("expected_region_count") is not None:
        checks["region_count_match"] = len(plan["open_regions"]) == int(
            case["expected_region_count"]
        )
    accepted_methods = [str(value) for value in case.get("accepted_operation_methods") or []]
    if accepted_methods:
        checks["operation_method_match"] = plan["operation_method"] in accepted_methods
    result["checks"] = checks
    result["passed"] = bool(checks) and all(checks.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate module 3 visual operation planning on historical simulator images."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--mode", choices=("disabled", "mock", "command", "http"))
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--protocol")
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = client_config_from_env(model=args.model)
    overrides = {
        key: value
        for key, value in {
            "mode": args.mode,
            "endpoint": args.endpoint,
            "model": args.model,
            "protocol": args.protocol,
            "timeout_s": args.timeout_s,
            "max_tokens": 256,
        }.items()
        if value is not None
    }
    config = replace(config, **overrides)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if not cases:
        raise ValueError("Evaluation manifest requires at least one case")
    client = MLLMClient(config)
    results = [_evaluate_case(client, manifest_path, case) for case in cases]
    report = {
        "module": "module3_visual_operation_planning",
        "model": config.model,
        "mode": config.mode,
        "protocol": config.protocol,
        "prompt": VISUAL_INTERACTION_PLANNING_INSTRUCTION,
        "case_count": len(results),
        "passed_count": sum(bool(result.get("passed")) for result in results),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(result.get("schema_valid") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
