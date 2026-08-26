#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced visual MLLM question bank from collected interaction samples."
    )
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attribute-cases", type=int, default=80)
    parser.add_argument("--max-verification-cases", type=int, default=40)
    parser.add_argument("--include-unknown-state", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def balanced_take(records: list[dict[str, Any]], limit: int, key_fn) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(key_fn(record))].append(record)
    for values in groups.values():
        values.sort(
            key=lambda item: (
                str(item.get("scene_id") or ""),
                -float(item.get("score", 0.0) or 0.0),
                str(item.get("object_id") or ""),
            )
        )
    selected = []
    while len(selected) < max(0, limit):
        progressed = False
        for key in sorted(groups):
            if not groups[key]:
                continue
            selected.append(groups[key].pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def attribute_case(index: int, record: dict[str, Any]) -> dict[str, Any]:
    parts = list(record.get("interaction_parts_gt") or [])
    expected = {
        "interactable": True,
        "interaction_class": str(record.get("interaction_class_gt") or "unknown"),
        "coarse_state": str(record.get("coarse_state_gt") or "unknown"),
    }
    return {
        "id": f"attribute_{index:04d}_{record['scene_id']}_{record['object_id']}",
        "module": "module1",
        "role": "attribute_inference",
        "instruction": "Infer visible interaction attributes for the cropped target object.",
        "context": {
            "object_id": record["object_id"],
            "name": record.get("semantic_name") or record.get("category") or "object",
            "category": record.get("category") or "",
            "scene_id": record.get("scene_id"),
        },
        "requires_image": True,
        "image_paths": [record["crop_path"]],
        "expected": expected,
        "metadata": {
            "house_ind": record.get("house_ind"),
            "step_index": record.get("step_index"),
            "source_object_name": record.get("source_object_name"),
            "crop": record.get("crop"),
            "joint_part_count_gt": len(parts),
        },
    }


def verification_case(index: int, record: dict[str, Any]) -> dict[str, Any] | None:
    previous_state = str(record.get("previous_state") or "unknown")
    current_state = str(record.get("coarse_state_gt") or "unknown")
    if previous_state == "closed" and current_state == "open":
        action = "open"
    elif previous_state == "open" and current_state == "closed":
        action = "close"
    else:
        return None
    return {
        "id": f"verify_{index:04d}_{record['scene_id']}_{record['object_id']}_{action}",
        "module": "module3",
        "role": "visual_verification",
        "instruction": "Inspect the current post-interaction crop and verify the requested action.",
        "context": {
            "object_id": record["object_id"],
            "previous_state": previous_state,
            "expected_state": current_state,
            "requested_action": action,
            "interaction_class": record.get("interaction_class_gt"),
        },
        "requires_image": True,
        "image_paths": [record["crop_path"]],
        "expected": {"success": True, "min_confidence": 0.5},
        "metadata": {
            "house_ind": record.get("house_ind"),
            "step_index": record.get("step_index"),
            "source_object_name": record.get("source_object_name"),
            "crop": record.get("crop"),
        },
    }


def main() -> None:
    args = parse_args()
    samples_dir = args.samples_dir.expanduser().resolve()
    attribute_records = read_jsonl(samples_dir / "attribute_samples.jsonl")
    verification_records = read_jsonl(samples_dir / "verification_samples.jsonl")
    if not args.include_unknown_state:
        attribute_records = [
            record
            for record in attribute_records
            if str(record.get("coarse_state_gt") or "unknown") != "unknown"
        ]
    attribute_records = balanced_take(
        attribute_records,
        args.max_attribute_cases,
        lambda item: (
            str(item.get("interaction_class_gt") or "unknown"),
            str(item.get("coarse_state_gt") or "unknown"),
        ),
    )
    verification_records = balanced_take(
        verification_records,
        args.max_verification_cases,
        lambda item: (
            str(item.get("interaction_class_gt") or "unknown"),
            str(item.get("previous_state") or "unknown"),
            str(item.get("coarse_state_gt") or "unknown"),
        ),
    )
    cases = [attribute_case(index, record) for index, record in enumerate(attribute_records)]
    cases.extend(
        case
        for index, record in enumerate(verification_records)
        if (case := verification_case(index, record)) is not None
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "description": "Realtime-GT interaction-object crop benchmark collected from exploration episodes.",
        "cases": cases,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "case_count": len(cases),
        "attribute_case_count": len(attribute_records),
        "verification_case_count": len(cases) - len(attribute_records),
        "scene_count": len({str(record.get("scene_id") or "") for record in attribute_records}),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
