#!/usr/bin/env python3
"""Build and run offline MLLM QA probes from a rule rollout.

Module 1 and Module 3 are image-grounded: each request receives one full RGB
frame with an anonymous target box.  Module 2 is deliberately text-only.  It
reconstructs the production ``ModelPolicyClient.build_request`` payload from
the recorded candidate/graph snapshot, so the evaluator tests subgoal ranking
rather than an unrelated candidate-to-pixel association.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "qwen3.6-35b-a3b-fp8"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
MODULE1_STAGE = "module1_visual_attribute"
MODULE2_STAGE = "module2_candidate_selection"
MODULE3_STAGE = "module3_visual_interaction_gate"


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


def is_visual_stage(stage: str) -> bool:
    return stage in {MODULE1_STAGE, MODULE3_STAGE}


def visual_prompt(stage: str, labels: list[str]) -> str:
    shared = (
        "图像是机器人当前完整第一视角；框只是匿名引用，不代表类别。不要使用图像外的"
        "场景知识、隐藏几何或对象名称。看不清、只看到侧面或没有可操作把手/门板时，"
        "优先选择 inspect 或 ignore，不能臆测可交互性。只返回 JSON。"
    )
    if stage == MODULE1_STAGE:
        return (
            f"{shared}\n"
            f"请判断 {labels[0]} 的可见属性，返回："
            '{"visible_object_type":"refrigerator|door|container|other|unknown",'
            '"view_relation":"front|side|back|oblique|unknown",'
            '"front_affordance_visible":true,"handle_visible":true,'
            '"confidence":0.0,"decision":"interact|inspect|ignore"}。'
        )
    if stage == MODULE2_STAGE:
        raise ValueError("Module 2 QA is text-only; use module2_text_prompt().")
    return (
        f"{shared}\n"
        f"请评估是否应立刻对 {labels[0]} 执行 open。返回："
        '{"decision":"interact|inspect|ignore","visible_object_type":'
        '"refrigerator|door|container|other|unknown",'
        '"front_affordance_visible":true,"handle_visible":true,'
        '"operation_method":"pull|push|slide|unknown",'
        '"open_region_in_target_bbox_normalized":[0.0,0.0],"confidence":0.0}。'
    )


def module2_http_context(request_payload: dict[str, Any]) -> dict[str, Any]:
    """Match the context passed by ``ModelPolicyClient._request_http`` exactly."""

    return {
        "mission": request_payload.get("mission") or {},
        "robot": request_payload.get("robot") or {},
        "graph": request_payload.get("graph") or {},
        "room_object_reasoning": request_payload.get("room_object_reasoning") or {},
        "candidates": request_payload.get("candidates") or [],
        "recent_decisions": request_payload.get("recent_decisions") or [],
    }


def module2_text_prompt(request_payload: dict[str, Any]) -> str:
    """Render the production Module-2 OpenAI-chat text request, without images."""

    instruction = str(request_payload.get("instruction") or "").rstrip()
    # The shipped local model configuration disables reasoning.  The shared
    # MLLM client consequently appends this marker before its HTTP call.
    if "/no_think" not in instruction:
        instruction += "\n/no_think"
    return instruction + "\n" + json.dumps(
        module2_http_context(request_payload), ensure_ascii=False
    )


def _load_runtime_model_policy() -> tuple[Any, Any, Any, Any]:
    """Load the exact production request builder without requiring ROS setup."""

    try:
        from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
        from semantic_decision_py_pkg.model_policy import (
            ModelPolicyClient,
            ModelPolicyConfig,
            aggregate_room_frontier_lengths,
        )
    except ModuleNotFoundError:
        repository_root = Path(__file__).resolve().parents[2]
        package_paths = (
            repository_root
            / "Interactive-Nav-SG-nav"
            / "src"
            / "semantic_decision_py_pkg"
            / "scripts",
            repository_root
            / "Interactive-Nav-SG-nav"
            / "src"
            / "semantic_mllm_py_pkg"
            / "scripts",
        )
        for package_path in reversed(package_paths):
            path_text = str(package_path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
        from semantic_decision_py_pkg.model_policy import (
            ModelPolicyClient,
            ModelPolicyConfig,
            aggregate_room_frontier_lengths,
        )
    return (
        BehaviorCandidate,
        ModelPolicyClient,
        ModelPolicyConfig,
        aggregate_room_frontier_lengths,
    )


def trace_for_candidate_sequence(
    traces: list[dict[str, Any]], candidate_sequence: int, selection_step: int
) -> dict[str, Any]:
    """Return the decision trace tied to the exact published candidate sequence."""

    matching = [
        event
        for event in traces
        if integer((event.get("payload") or {}).get("input_candidate_sequence"))
        == candidate_sequence
    ]
    if not matching:
        return {}
    return min(
        matching,
        key=lambda event: (
            abs(integer(event.get("step_id")) - selection_step),
            integer(event.get("step_id")),
        ),
    )


def source_candidate_event(
    candidates_by_step: list[dict[str, Any]],
    selected_id: str,
    selection_step: int,
    candidate_sequence: int,
) -> dict[str, Any]:
    """Find the candidate snapshot that produced one selected event."""

    for event in reversed(candidates_by_step):
        payload = event.get("payload") or {}
        if integer(event.get("step_id")) > selection_step:
            continue
        if candidate_sequence >= 0 and integer(payload.get("sequence")) == candidate_sequence:
            return event
    for event in reversed(candidates_by_step):
        payload = event.get("payload") or {}
        if integer(event.get("step_id")) > selection_step:
            continue
        values = list(payload.get("candidates") or [])
        if any(
            str(item.get("candidate_id") or "") == selected_id
            for item in values
            if isinstance(item, dict)
        ):
            return event
    return {}


def build_module2_request(
    candidate_snapshot: dict[str, Any],
    trace_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the real Module-2 request from a recorded decision snapshot.

    The record contains the raw candidate snapshot and the curator trace.  The
    latter identifies the curated candidate IDs and score/hint inputs used by
    the live decision node.  Histories absent from the recorder remain empty;
    this limitation is surfaced in the case provenance rather than replaced by
    image-derived guesses.
    """

    (
        BehaviorCandidate,
        ModelPolicyClient,
        ModelPolicyConfig,
        aggregate_room_frontier_lengths,
    ) = _load_runtime_model_policy()
    raw_candidates = [
        item
        for item in list(candidate_snapshot.get("candidates") or [])
        if isinstance(item, dict)
    ]
    by_id = {
        str(candidate.get("candidate_id") or ""): candidate for candidate in raw_candidates
    }
    curation = trace_payload.get("candidate_curation") or {}
    curated_ids = [
        str(candidate_id)
        for candidate_id in list(curation.get("selected_ids") or [])
        if str(candidate_id) in by_id
    ]
    selected_raw_candidates = (
        [by_id[candidate_id] for candidate_id in curated_ids]
        if curated_ids
        else raw_candidates
    )
    all_candidates = [BehaviorCandidate(**candidate) for candidate in raw_candidates]
    model_candidates = [
        BehaviorCandidate(**candidate) for candidate in selected_raw_candidates
    ]
    graph = dict(candidate_snapshot.get("graph_context") or {})
    selection_granularity = str(
        trace_payload.get("model_selection_granularity") or "candidate"
    )
    client = ModelPolicyClient(
        ModelPolicyConfig(mode="disabled", selection_granularity=selection_granularity)
    )
    robot_context = {
        "robot_xy": candidate_snapshot.get("robot_xy"),
        "exploration_context": candidate_snapshot.get("exploration_context") or {},
        "active_candidate_id": str(trace_payload.get("active_candidate_id") or ""),
        "decision_history": list(trace_payload.get("recent_decisions") or []),
        "group_history": list(trace_payload.get("group_history") or []),
        "candidate_history": dict(trace_payload.get("candidate_history") or {}),
        "room_frontier_lengths": aggregate_room_frontier_lengths(all_candidates, graph),
        "candidate_pre_scores": dict(curation.get("quality_by_id") or {}),
        "candidate_pre_score_terms": dict(curation.get("quality_terms_by_id") or {}),
        "candidate_decision_hints": dict(curation.get("decision_hint_by_id") or {}),
        "entered_room_ids": list(curation.get("entered_room_ids") or []),
    }
    request = client.build_request(
        model_candidates,
        dict(candidate_snapshot.get("target_context") or {}),
        graph,
        robot_context,
    )
    provenance = {
        "request_builder": "semantic_decision_py_pkg.model_policy.ModelPolicyClient.build_request",
        "selection_granularity": selection_granularity,
        "candidate_pool_source": "candidate_curation.selected_ids" if curated_ids else "snapshot.candidates",
        "raw_candidate_count": len(raw_candidates),
        "model_candidate_count": len(model_candidates),
        "recorded_history_fields": {
            "decision_history": "recent_decisions",
            "group_history": "group_history" if trace_payload.get("group_history") else "not_recorded",
            "candidate_history": "candidate_history" if trace_payload.get("candidate_history") else "not_recorded",
        },
    }
    return request, provenance


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


def selected_event_cases(events: list[dict[str, Any]], frames: list[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_step = [
        event for event in events if event.get("type") == "semantic_decision_candidates"
    ]
    traces = [event for event in events if event.get("type") == "semantic_decision_trace"]
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
        candidate_sequence = integer(payload.get("candidate_sequence"))
        candidate_event = source_candidate_event(
            candidates_by_step, selected_id, step, candidate_sequence
        )
        candidate_snapshot = candidate_event.get("payload") or {}
        if not candidate_snapshot:
            continue
        trace_event = trace_for_candidate_sequence(traces, candidate_sequence, step)
        trace_payload = trace_event.get("payload") or {}
        try:
            request, request_provenance = build_module2_request(
                candidate_snapshot, trace_payload
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"failed to reconstruct text-only Module-2 request at step {step}: {exc}"
            ) from exc
        if request.get("candidates"):
            module2.append(
                {
                    "module2_request": request,
                    "request_provenance": request_provenance,
                    "selection_step": step,
                    "candidate_sequence": candidate_sequence,
                    "heldout": {
                        "rule_reference_selected_candidate_id": selected_id,
                        "selection_step": step,
                        "candidate_sequence": candidate_sequence,
                    },
                }
            )
        if str(payload.get("behavior_type") or "") != "INTERACT":
            continue
        frame = nearest_fresh_frame(frames, step)
        if frame is None:
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
    if not is_visual_stage(stage):
        raise ValueError(f"{stage} is not an image-grounded QA stage")
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
        cases.append(
            {
                "case_id": case_id,
                "stage": stage,
                "image_path": str(annotated_path),
                "raw_full_frame_path": frame["frame"],
                "prompt": visual_prompt(stage, labels),
                "public_context": public_context,
                "expected_heldout": heldout,
                "provenance": {"rule_run_dir": str(output_dir.parent)},
            }
        )
    return cases


def materialize_module2_cases(
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit production-schema Module-2 cases without frames, boxes, or images."""

    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        request = dict(record.get("module2_request") or {})
        if not request:
            continue
        case_id = f"{MODULE2_STAGE}_{index:03d}"
        candidates = list(request.get("candidates") or [])
        cases.append(
            {
                "case_id": case_id,
                "stage": MODULE2_STAGE,
                "modality": "text",
                "prompt": module2_text_prompt(request),
                "module2_request": request,
                "public_context": {
                    "selection_step": integer(record.get("selection_step")),
                    "candidate_sequence": integer(record.get("candidate_sequence")),
                    "request_schema_version": integer(request.get("schema_version")),
                    "candidate_count": len(candidates),
                    "candidate_ids": [
                        str(candidate.get("id") or "") for candidate in candidates
                    ],
                },
                "expected_heldout": dict(record.get("heldout") or {}),
                "provenance": {
                    "rule_run_dir": str(output_dir.parent),
                    **dict(record.get("request_provenance") or {}),
                },
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


def _html_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def render_html_report(
    output_dir: Path,
    cases: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> Path:
    """Write a portable QA page; only visual stages contain an image element."""

    responses_by_case = {
        str(response.get("case_id") or ""): response for response in responses
    }
    cards: list[str] = []
    for case in cases:
        case_id = str(case.get("case_id") or "unknown")
        stage = str(case.get("stage") or "")
        response = responses_by_case.get(case_id, {})
        title = {
            MODULE1_STAGE: "M1 · visual attribute",
            MODULE2_STAGE: "M2 · text-only subgoal ranking",
            MODULE3_STAGE: "M3 · visual interaction gate",
        }.get(stage, stage)
        if is_visual_stage(stage):
            image_path = Path(str(case.get("image_path") or ""))
            try:
                image_src = image_path.resolve().relative_to(output_dir.resolve()).as_posix()
            except ValueError:
                image_src = str(image_path)
            media = (
                '<figure><img src="'
                + html.escape(image_src, quote=True)
                + '" alt="'
                + html.escape(case_id, quote=True)
                + '"><figcaption>Anonymous visual reference only.</figcaption></figure>'
            )
        else:
            media = (
                '<aside class="text-only"><strong>No image is sent to M2.</strong>'
                '<br>This card contains the production-format graph/candidate request only.</aside>'
            )
        result = response.get("parsed_response")
        if result is None:
            result = {"error": response.get("error") or "no parsed model response"}
        latency = response.get("latency_sec")
        latency_text = "" if latency is None else f" · {float(latency):.3f}s"
        cards.append(
            "<article class=\"case\" id=\""
            + html.escape(case_id, quote=True)
            + "\"><header><h2>"
            + html.escape(title)
            + "</h2><span>"
            + html.escape(case_id + latency_text)
            + "</span></header><div class=\"grid\">"
            + media
            + "<section><h3>Request</h3><details"
            + (" open" if stage == MODULE2_STAGE else "")
            + "><summary>Show prompt / payload</summary><pre>"
            + html.escape(str(case.get("prompt") or ""))
            + "</pre></details><h3>Model JSON</h3><pre>"
            + _html_json(result)
            + "</pre></section></div></article>"
        )
    page = """<!doctype html>
<html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Interactive Navigation MLLM QA</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}main{max-width:1360px;margin:28px auto;padding:0 20px}.note{color:#b8c8dc}.case{border-top:1px solid #445;padding:22px 0}header{display:flex;justify-content:space-between;gap:14px}h2{margin:0;font-size:1.1rem}h3{margin:0 0 8px;font-size:.92rem}.grid{display:grid;grid-template-columns:minmax(260px,1.1fr) minmax(320px,.9fr);gap:18px;margin-top:14px}figure{margin:0}img{display:block;width:100%;border-radius:6px}figcaption,header span{color:#aab;font-size:.84rem;margin-top:6px}.text-only{border:1px dashed #6284aa;background:#162333;border-radius:6px;padding:18px;line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c1c1c;padding:12px;border-radius:6px;margin:0;font-size:.78rem}summary{cursor:pointer;margin-bottom:8px}@media(max-width:760px){.grid{grid-template-columns:1fr}header{display:block}header span{display:block}}
</style><main><h1>Interactive Navigation MLLM QA</h1><p class=\"note\">M1/M3 are image-grounded. M2 reconstructs the production graph/candidate request and sends text only—no RGB, bbox, or image URL. This report reuses a pre-fix rollout; its visual boxes are archival and do not validate the new GT bbox implementation.</p>""" + "".join(cards) + "</main></html>"
    report_path = output_dir / "qa_results.html"
    report_path.write_text(page, encoding="utf-8")
    return report_path


def call_model(case: dict[str, Any], args: argparse.Namespace, client: Any) -> dict[str, Any]:
    started = time.perf_counter()
    if case["stage"] == MODULE2_STAGE:
        messages = [
            {"role": "system", "content": "Return only a valid JSON object."},
            {
                "role": "user",
                "content": [{"type": "text", "text": case["prompt"]}],
            },
        ]
        # The local vLLM endpoint consumes this chat-template control.  Keep
        # the text payload identical to runtime M2 while ensuring it returns
        # the requested compact JSON rather than an echoed context.
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": case["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(Path(case["image_path"]))},
                    },
                ],
            }
        ]
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        response_format={"type": "json_object"},
        extra_body=extra_body,
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
        materialize_cases(output_dir, MODULE1_STAGE, stage1_records)
        + materialize_module2_cases(output_dir, stage2_records)
        + materialize_cases(output_dir, MODULE3_STAGE, stage3_records)
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
    html_report_path = render_html_report(output_dir, cases, responses)
    by_stage: dict[str, int] = defaultdict(int)
    for case in cases:
        by_stage[case["stage"]] += 1
    unsafe_interact = sum(
        1
        for response in responses
        if isinstance(response.get("parsed_response"), dict)
        and response["parsed_response"].get("decision") == "interact"
        and response["stage"] in {MODULE1_STAGE, MODULE3_STAGE}
        and not bool(response["parsed_response"].get("front_affordance_visible", False))
    )
    summary = {
        "run_dir": str(run_dir),
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "visual_stages_full_frame_only": True,
        "module2_text_only": True,
        "stage_modalities": {
            MODULE1_STAGE: "full_frame_image",
            MODULE2_STAGE: "text",
            MODULE3_STAGE: "full_frame_image",
        },
        "case_count": len(cases),
        "case_count_by_stage": dict(by_stage),
        "response_count": len(responses),
        "unsafe_self_reported_interact_count": unsafe_interact,
        "html_report": str(html_report_path),
        "notes": [
            "GT names, AABBs, joint axes, and rule selections are held out from visual prompts.",
            "Module 2 uses the production ModelPolicyClient build_request schema and receives no image, bbox, or pixel-derived candidate context.",
            "Fresh image/bbox pairs require image_step == observation_capture_step.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
