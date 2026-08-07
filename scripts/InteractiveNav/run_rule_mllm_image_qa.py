#!/usr/bin/env python3
"""Build and run full-frame, bbox-grounded QA probes from a rule rollout.

The script deliberately keeps simulator labels and oracle geometry in held-out
provenance.  The model receives exactly one image per request: the original
full RGB frame with anonymous target/candidate boxes drawn on it.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "qwen3.6-35b-a3b-fp8"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--cases-per-stage", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero makes an offline QA record reproducible.",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def valid_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x0, y0, x1, y1 = (integer(item) for item in value[:4])
    left, right = sorted((max(0, x0), min(width - 1, x1)))
    top, bottom = sorted((max(0, y0), min(height - 1, y1)))
    if right - left < 4 or bottom - top < 4:
        return None
    return [left, top, right, bottom]


def fresh_frames(run_dir: Path) -> list[dict[str, Any]]:
    manifest = run_dir / "sim_step_frames" / "manifest.jsonl"
    frames: list[dict[str, Any]] = []
    for row in read_jsonl(manifest):
        step = integer(row.get("step_index"))
        frame = Path(str(row.get("frame") or ""))
        gt = row.get("gt_observations") or {}
        if integer(gt.get("capture_step")) != step or not frame.is_file():
            continue
        observations = [
            item for item in (gt.get("observations") or []) if isinstance(item, dict)
        ]
        if not observations:
            continue
        frames.append(
            {
                "step": step,
                "frame": str(frame),
                "width": integer(row.get("width"), 1024),
                "height": integer(row.get("height"), 576),
                "observations": observations,
                "capture_step": integer(gt.get("capture_step")),
            }
        )
    return sorted(frames, key=lambda item: item["step"])


def nearest_fresh_frame(frames: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    if not frames:
        return None
    return min(frames, key=lambda item: (abs(item["step"] - step), item["step"]))


def observation_for_target(frame: dict[str, Any], target_id: str) -> dict[str, Any] | None:
    target = str(target_id or "")
    for observation in frame.get("observations") or []:
        if str(observation.get("id") or "") == target:
            bbox = valid_bbox(observation.get("bbox_2d"), frame["width"], frame["height"])
            if bbox is not None:
                return {**observation, "bbox_2d": bbox}
    return None


def observation_kind(observation: dict[str, Any]) -> str:
    text = " ".join(
        str(observation.get(key) or "")
        for key in ("id", "name", "semantic_name", "category")
    ).casefold()
    if "refrigerator" in text or "fridge" in text:
        return "refrigerator"
    if "door" in text or "portal" in text:
        return "door"
    if any(token in text for token in ("cabinet", "drawer", "dresser", "container")):
        return "container"
    return "other"


def evenly(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indexes]


def draw_full_frame(
    source: Path,
    destination: Path,
    annotations: list[tuple[str, list[int]]],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    colors = [(220, 40, 220), (20, 190, 255), (255, 170, 0), (20, 220, 130)]
    for index, (label, bbox) in enumerate(annotations):
        color = colors[index % len(colors)]
        draw.rectangle(tuple(bbox), outline=color, width=4)
        left, top = bbox[0], max(0, bbox[1] - 24)
        text_box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((left, top), label, fill=color, font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def visual_prompt(stage: str, labels: list[str], candidate_context: list[dict[str, Any]]) -> str:
    shared = (
        "图像是机器人当前完整第一视角；框只是匿名引用，不代表类别。不要使用图像外的"
        "场景知识、隐藏几何或对象名称。看不清、只看到侧面或没有可操作把手/门板时，"
        "优先选择 inspect 或 ignore，不能臆测可交互性。只返回 JSON。"
    )
    if stage == "module1_visual_attribute":
        return (
            f"{shared}\n"
            f"请判断 {labels[0]} 的可见属性，返回："
            '{"visible_object_type":"refrigerator|door|container|other|unknown",'
            '"view_relation":"front|side|back|oblique|unknown",'
            '"front_affordance_visible":true,"handle_visible":true,'
            '"confidence":0.0,"decision":"interact|inspect|ignore"}。'
        )
    if stage == "module2_candidate_selection":
        return (
            f"{shared}\n候选摘要如下（名称已匿名）：\n"
            f"{json.dumps(candidate_context, ensure_ascii=False)}\n"
            '请返回 {"ranked_candidates":["cand_A"],"decision":"select|inspect|defer",'
            '"confidence":0.0,"reason":"简短原因"}。'
        )
    return (
        f"{shared}\n"
        f"请评估是否应立刻对 {labels[0]} 执行 open。返回："
        '{"decision":"interact|inspect|ignore","visible_object_type":'
        '"refrigerator|door|container|other|unknown",'
        '"front_affordance_visible":true,"handle_visible":true,'
        '"operation_method":"pull|push|slide|unknown",'
        '"open_region_in_target_bbox_normalized":[0.0,0.0],"confidence":0.0}。'
    )


def heldout_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_target_id": str(observation.get("id") or ""),
        "raw_target_name": str(observation.get("name") or ""),
        "selection_kind": observation_kind(observation),
        "bbox_xyxy": list(observation.get("bbox_2d") or []),
        "visible_pixels": integer(observation.get("visible_pixels"), 0),
        "visible_fraction": number(observation.get("visible_fraction"), 0.0),
    }


def build_module1_cases(frames: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for observation in frame["observations"]:
            bbox = valid_bbox(observation.get("bbox_2d"), frame["width"], frame["height"])
            if bbox is None:
                continue
            kind = observation_kind(observation)
            if kind in {"refrigerator", "door", "container"}:
                by_kind[kind].append({"frame": frame, "observation": {**observation, "bbox_2d": bbox}})
    # A stationary robot can emit the identical full RGB/bbox pair at multiple
    # fresh observation steps.  Keep one such pair so a QA batch measures
    # visual diversity rather than sampling variance on a duplicate prompt.
    preferred: list[dict[str, Any]] = []
    for kind in ("refrigerator", "door", "container"):
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for record in by_kind[kind]:
            observation = record["observation"]
            signature = (
                str(observation.get("id") or ""),
                tuple(integer(value) for value in observation.get("bbox_2d") or []),
            )
            if signature in seen:
                continue
            seen.add(signature)
            preferred.append(record)
    return evenly(preferred, count)


def candidate_summary(candidate: dict[str, Any], label: str) -> dict[str, Any]:
    metadata = candidate.get("metadata") or {}
    features = candidate.get("features") or {}
    return {
        "id": label,
        "behavior": str(candidate.get("behavior_type") or "unknown"),
        "distance_m": round(number(features.get("distance_m")), 2),
        "interaction_cost": round(number(features.get("interaction_cost")), 2),
        "expected_effect": str(metadata.get("expected_effect") or "unknown"),
        "state": str(metadata.get("state") or "unknown"),
        "currently_visible": bool(metadata.get("is_currently_visible", False)),
    }


def selected_event_cases(events: list[dict[str, Any]], frames: list[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_step = [event for event in events if event.get("type") == "semantic_decision_candidates"]
    selected = [
        event
        for event in events
        if event.get("type") == "semantic_decision_selected"
        and str((event.get("payload") or {}).get("behavior_type") or "") != "SCAN"
        and bool((event.get("payload") or {}).get("active", False))
    ]
    module2: list[dict[str, Any]] = []
    module3: list[dict[str, Any]] = []
    for event in selected:
        payload = event.get("payload") or {}
        selected_id = str(payload.get("candidate_id") or "")
        if not selected_id:
            continue
        step = integer(event.get("step_id"))
        frame = nearest_fresh_frame(frames, step)
        if frame is None:
            continue
        source_candidates: list[dict[str, Any]] = []
        for candidate_event in reversed(candidates_by_step):
            if integer(candidate_event.get("step_id")) > step:
                continue
            values = list((candidate_event.get("payload") or {}).get("candidates") or [])
            if any(str(item.get("candidate_id") or "") == selected_id for item in values if isinstance(item, dict)):
                source_candidates = [item for item in values if isinstance(item, dict)]
                break
        if not source_candidates:
            continue
        visible: list[tuple[str, dict[str, Any]]] = []
        labeled: list[dict[str, Any]] = []
        reference_label = ""
        for index, candidate in enumerate(source_candidates[:4]):
            label = f"cand_{chr(ord('A') + index)}"
            target_id = str(candidate.get("target_id") or "")
            observation = observation_for_target(frame, target_id)
            if observation is not None:
                visible.append((label, observation))
            labeled.append(candidate_summary(candidate, label))
            if str(candidate.get("candidate_id") or "") == selected_id:
                reference_label = label
        if visible and reference_label:
            module2.append(
                {
                    "frame": frame,
                    "annotations": visible,
                    "candidate_context": labeled,
                    "heldout": {
                        "rule_reference_selected_candidate": reference_label,
                        "raw_rule_candidate_id": selected_id,
                        "selection_step": step,
                    },
                }
            )
        if str(payload.get("behavior_type") or "") != "INTERACT":
            continue
        target = observation_for_target(frame, str(payload.get("target_id") or ""))
        if target is None:
            continue
        module3.append(
            {
                "frame": frame,
                "annotations": [("target_01", target)],
                "heldout": {
                    **heldout_observation(target),
                    "raw_rule_candidate_id": selected_id,
                    "selection_step": step,
                    "rule_reference_behavior": "INTERACT",
                },
            }
        )
    return evenly(module2, count), evenly(module3, count)


def materialize_cases(
    output_dir: Path,
    stage: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        frame = record["frame"]
        annotations = record.get("annotations")
        if annotations is None:
            observation = record["observation"]
            annotations = [("target_01", observation)]
            heldout = heldout_observation(observation)
        else:
            heldout = dict(record.get("heldout") or {})
        normalized_annotations = [
            (label, list(observation.get("bbox_2d") or observation))
            for label, observation in annotations
        ]
        case_id = f"{stage}_{index:03d}"
        annotated_path = output_dir / "annotated_frames" / f"{case_id}.png"
        draw_full_frame(Path(frame["frame"]), annotated_path, normalized_annotations)
        labels = [label for label, _ in normalized_annotations]
        public_context = {
            "image_step": frame["step"],
            "observation_capture_step": frame["capture_step"],
            "capture_lag_steps": frame["step"] - frame["capture_step"],
            "frame_size": [frame["width"], frame["height"]],
            "boxes": [{"label": label, "bbox_xyxy": bbox} for label, bbox in normalized_annotations],
        }
        candidate_context = list(record.get("candidate_context") or [])
        cases.append(
            {
                "case_id": case_id,
                "stage": stage,
                "image_path": str(annotated_path),
                "raw_full_frame_path": frame["frame"],
                "prompt": visual_prompt(stage, labels, candidate_context),
                "public_context": {**public_context, "candidates": candidate_context},
                "expected_heldout": heldout,
                "provenance": {"rule_run_dir": str(output_dir.parent)},
            }
        )
    return cases


def parse_response(content: str) -> tuple[dict[str, Any] | None, str]:
    text = str(content or "").strip()
    try:
        value = json.loads(text)
        return (value if isinstance(value, dict) else None), text
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return (value if isinstance(value, dict) else None), text
            except json.JSONDecodeError:
                pass
    return None, text


def call_model(case: dict[str, Any], args: argparse.Namespace, client: Any) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case["prompt"]},
                    {"type": "image_url", "image_url": {"url": data_url(Path(case["image_path"]))}},
                ],
            }
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = str(response.choices[0].message.content or "")
    parsed, raw = parse_response(content)
    usage = response.usage.model_dump() if response.usage is not None else {}
    return {
        "case_id": case["case_id"],
        "stage": case["stage"],
        "latency_sec": time.perf_counter() - started,
        "raw_response": raw,
        "parsed_response": parsed,
        "usage": usage,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "mllm_image_qa").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = fresh_frames(run_dir)
    events = read_jsonl(run_dir / "debug" / "events.jsonl")
    stage1_records = build_module1_cases(frames, args.cases_per_stage)
    stage2_records, stage3_records = selected_event_cases(
        events, frames, args.cases_per_stage
    )
    cases = (
        materialize_cases(output_dir, "module1_visual_attribute", stage1_records)
        + materialize_cases(output_dir, "module2_candidate_selection", stage2_records)
        + materialize_cases(output_dir, "module3_visual_interaction_gate", stage3_records)
    )
    write_jsonl(output_dir / "cases.jsonl", cases)
    responses: list[dict[str, Any]] = []
    if not args.skip_model:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            http_client=httpx.Client(trust_env=False, timeout=args.timeout_s),
        )
        for case in cases:
            try:
                responses.append(call_model(case, args, client))
            except Exception as exc:
                responses.append(
                    {
                        "case_id": case["case_id"],
                        "stage": case["stage"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        client.close()
    write_jsonl(output_dir / "responses.jsonl", responses)
    by_stage: dict[str, int] = defaultdict(int)
    for case in cases:
        by_stage[case["stage"]] += 1
    unsafe_interact = sum(
        1
        for response in responses
        if isinstance(response.get("parsed_response"), dict)
        and response["parsed_response"].get("decision") == "interact"
        and response["stage"] in {"module1_visual_attribute", "module3_visual_interaction_gate"}
        and not bool(response["parsed_response"].get("front_affordance_visible", False))
    )
    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "full_frame_only": True,
        "case_count": len(cases),
        "case_count_by_stage": dict(by_stage),
        "response_count": len(responses),
        "unsafe_self_reported_interact_count": unsafe_interact,
        "notes": [
            "GT names, AABBs, joint axes, and rule selections are held out from visual prompts.",
            "Module-2 labels are rule references, not visual ground truth.",
            "Fresh image/bbox pairs require image_step == observation_capture_step.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
