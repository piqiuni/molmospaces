"""Generate rule-, LLM-, or VLM-authored InstructionGoal V3 episodes."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.interactive_nav_grounded_plan import (
    build_grounded_plan,
    build_path_corridor_graph,
    load_episodes,
    select_episode,
    select_segment_keyframes,
)


DISCLOSURES = ("hidden", "partial", "explicit")


def _humanize_label(value: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(value or ""))
    return re.sub(r"[_-]+", " ", text).strip().lower()


def _target_phrase(plan: dict[str, Any]) -> str:
    target = plan["target"]
    if target["target_type"] == "point":
        return "the designated point"
    label = target.get("referral_expression") or target.get("category") or "target"
    return f"the {_humanize_label(label)}"


def _interaction_entities(plan: dict[str, Any]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for step in plan["steps"]:
        if step.get("type") != "open_joint":
            continue
        interaction = step.get("interaction") or {}
        name = str(step.get("object_name") or interaction.get("object_name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        category = str(interaction.get("object_category") or "interactive object")
        entities.append(
            {
                "object_name": name,
                "object_category": category,
                "interaction_type": str(interaction.get("type") or ""),
            }
        )
    return entities


def rule_instruction(plan: dict[str, Any], disclosure: str) -> dict[str, Any]:
    if disclosure not in DISCLOSURES:
        raise ValueError(f"Unsupported interaction disclosure: {disclosure}")
    target_phrase = _target_phrase(plan)
    entities = _interaction_entities(plan)
    if disclosure == "hidden" or not entities:
        text = (
            f"Navigate to {target_phrase}."
            if plan["target"]["target_type"] == "point"
            else f"Find {target_phrase}."
        )
    elif disclosure == "partial":
        text = (
            f"Follow the route to {target_phrase}, interacting with blocked passages "
            "or enclosed spaces if needed."
        )
    else:
        clauses = []
        for entity in entities:
            category = _humanize_label(entity["object_category"])
            interaction_type = entity["interaction_type"]
            if interaction_type.startswith("channel_"):
                clauses.append(f"go to the {category} and open it")
            elif interaction_type.startswith("container_"):
                clauses.append(f"continue to the {category} and open it")
            else:
                clauses.append(f"interact with the {category}")
        terminal = (
            f"continue to {target_phrase}"
            if plan["target"]["target_type"] == "point"
            else f"find {target_phrase}"
        )
        text = ", then ".join(clauses + [terminal]).capitalize() + "."
    grounded_ids = (
        [entity["object_name"] for entity in entities]
        if disclosure == "explicit"
        else []
    )
    target = plan["target"]
    if target.get("selected_instance"):
        grounded_ids.append(str(target["selected_instance"]))
    grounded_steps = [
        int(step["step_index"])
        for step in plan["steps"]
        if step.get("type") in {"navigate", "open_joint", "observe_target"}
    ]
    return {
        "instruction": text,
        "instruction_type": (
            "route_interaction_instruction" if entities else "route_instruction"
        ),
        "interaction_disclosure": disclosure,
        "grounded_entity_ids": list(dict.fromkeys(grounded_ids)),
        "grounded_plan_step_indices": grounded_steps,
        "generator": "rule",
    }


def _valid_entity_ids(plan: dict[str, Any], graph_context: dict[str, Any] | None) -> set[str]:
    valid: set[str] = set()
    target = plan["target"]
    for key in ("selected_instance", "container_name"):
        if target.get(key):
            valid.add(str(target[key]))
    for entity in _interaction_entities(plan):
        valid.add(entity["object_name"])
    for node in (graph_context or {}).get("nodes") or []:
        for value in (
            node.get("id"),
            node.get("name"),
            (node.get("attributes") or {}).get("source_object_name"),
        ):
            if value:
                valid.add(str(value))
    return valid


def validate_model_instruction(
    result: dict[str, Any],
    *,
    plan: dict[str, Any],
    disclosure: str,
    graph_context: dict[str, Any] | None,
) -> dict[str, Any]:
    instruction = str(result.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("Instruction model returned an empty instruction")
    returned_disclosure = str(result.get("interaction_disclosure") or disclosure)
    if returned_disclosure != disclosure:
        raise ValueError("Instruction model changed the requested disclosure level")
    grounded_ids = [str(value) for value in result.get("grounded_entity_ids") or []]
    unknown = set(grounded_ids) - _valid_entity_ids(plan, graph_context)
    if unknown:
        raise ValueError(f"Instruction model grounded unknown entities: {sorted(unknown)}")
    valid_steps = {int(step["step_index"]) for step in plan["steps"]}
    grounded_steps = [int(value) for value in result.get("grounded_plan_step_indices") or []]
    if not set(grounded_steps).issubset(valid_steps):
        raise ValueError("Instruction model referenced an unknown plan step")
    instruction_type = str(
        result.get("instruction_type")
        or ("route_interaction_instruction" if _interaction_entities(plan) else "route_instruction")
    )
    allowed_types = {
        "object_goal",
        "point_goal",
        "route_instruction",
        "interaction_instruction",
        "route_interaction_instruction",
    }
    if instruction_type not in allowed_types:
        raise ValueError(f"Unsupported generated instruction_type: {instruction_type}")
    return {
        "instruction": instruction,
        "instruction_type": instruction_type,
        "interaction_disclosure": disclosure,
        "grounded_entity_ids": grounded_ids,
        "grounded_plan_step_indices": grounded_steps,
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "evidence_frame_indices": [
            int(value) for value in result.get("evidence_frame_indices") or []
        ],
        "generator": str(result.get("generator") or "model"),
    }


def apply_instruction(
    source_episode: dict[str, Any],
    generated: dict[str, Any],
    *,
    generation_mode: str,
    graph_context: dict[str, Any] | None = None,
    model_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    episode = copy.deepcopy(source_episode)
    language = copy.deepcopy(episode.get("language") or {})
    language.update(
        {
            "task_description": generated["instruction"],
            "instruction_type": generated["instruction_type"],
            "interaction_disclosure": generated["interaction_disclosure"],
            "locale": language.get("locale") or "en",
            "task_input_mode": "instruction",
            "grounded_entity_ids": generated["grounded_entity_ids"],
            "grounded_plan_step_indices": generated["grounded_plan_step_indices"],
            "generation_mode": generation_mode,
        }
    )
    episode["language"] = language
    interactive = episode.setdefault("interactive_nav", {})
    source_case_id = str(interactive.get("case_id") or "episode")
    suffix = generated["interaction_disclosure"]
    interactive["case_id"] = f"{source_case_id}__instruction_{generation_mode}_{suffix}"
    interactive["instruction_generation"] = {
        "schema_version": "interactive_nav_instruction_generation_v1",
        "source_case_id": source_case_id,
        "mode": generation_mode,
        "grounded_entity_ids": generated["grounded_entity_ids"],
        "grounded_plan_step_indices": generated["grounded_plan_step_indices"],
        "evidence_frame_indices": generated.get("evidence_frame_indices", []),
        "graph_context": graph_context,
        "model_metrics": model_metrics,
    }
    return v3.validate_instruction_goal_v3_episode(episode)


def _all_path_waypoints(plan: dict[str, Any]) -> list[list[float]]:
    if plan.get("gt_path_waypoints"):
        return [list(point) for point in plan["gt_path_waypoints"]]
    output: list[list[float]] = []
    for step in plan["steps"]:
        for point in step.get("path_waypoints") or []:
            if not output or output[-1] != point:
                output.append(point)
    if not output:
        output = [
            list(step["goal_point"])
            for step in plan["steps"]
            if step.get("type") == "navigate" and len(step.get("goal_point") or []) >= 2
        ]
    return output


def load_graph_context(
    graph_path: Path | None,
    *,
    plan: dict[str, Any],
    radius_m: float,
) -> dict[str, Any] | None:
    if graph_path is None:
        return None
    graph = json.loads(graph_path.read_text())
    required = list(plan.get("required_interaction_ids") or [])
    required.extend(entity["object_name"] for entity in _interaction_entities(plan))
    target = plan["target"]
    if target.get("selected_instance"):
        required.append(target["selected_instance"])
    return build_path_corridor_graph(
        graph,
        _all_path_waypoints(plan),
        radius_m=radius_m,
        required_entity_ids=required,
    )


def extract_full_rollout_keyframes(
    trajectory: Path,
    output_dir: Path,
    *,
    camera_name: str = "head_camera",
) -> tuple[list[Path], list[int]]:
    import h5py
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(trajectory, "r") as handle:
        steps = handle["steps"]
        segments = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in steps["segment"][:]
        ]
        if camera_name not in steps["images"]:
            raise KeyError(f"Camera {camera_name!r} is missing from {trajectory}")
        indices = select_segment_keyframes(segments)
        images = steps["images"][camera_name]
        paths = []
        for index in indices:
            path = output_dir / f"step_{index:06d}_{segments[index]}.png"
            Image.fromarray(images[index]).save(path)
            paths.append(path)
    return paths, indices


def _mllm_import_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts"
    )


def model_instruction(
    *,
    plan: dict[str, Any],
    disclosure: str,
    graph_context: dict[str, Any] | None,
    images: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.model_mode == "mock":
        generated = rule_instruction(plan, disclosure)
        generated["generator"] = "model_mock"
        if images:
            generated["evidence_frame_indices"] = list(range(len(images)))
        return generated, {"mode": "mock"}
    package_path = _mllm_import_path()
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig

    client = MLLMClient(
        MLLMClientConfig(
            mode=args.model_mode,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            model=args.model,
            protocol=args.protocol,
            command=args.command,
            timeout_s=args.timeout_s,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            image_detail=args.image_detail,
        )
    )
    context = {
        "grounded_plan": plan,
        "path_corridor_graph": graph_context or {},
        "requested_disclosure": disclosure,
        "constraints": {
            "do_not_invent_entities": True,
            "preserve_action_order": True,
            "return_grounding": True,
        },
    }
    response = client.request_json(
        role="instruction_generation",
        instruction=(
            "Generate one concise natural-language navigation instruction from the "
            "grounded plan. Preserve action order, use only supplied entities and return "
            "instruction, instruction_type, interaction_disclosure, grounded_entity_ids, "
            "grounded_plan_step_indices, confidence, and optional evidence_frame_indices."
        ),
        context=context,
        images=[str(path) for path in images],
        response_schema={
            "type": "object",
            "required": [
                "instruction",
                "instruction_type",
                "interaction_disclosure",
                "grounded_entity_ids",
                "grounded_plan_step_indices",
            ],
        },
    )
    if response.error or not isinstance(response.payload, dict):
        raise RuntimeError(response.error or "Instruction model returned no JSON payload")
    generated = validate_model_instruction(
        response.payload,
        plan=plan,
        disclosure=disclosure,
        graph_context=graph_context,
    )
    generated["generator"] = "vlm" if images else "llm"
    return generated, response.metrics()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["rule", "llm", "vlm"], default="rule")
    parser.add_argument("--disclosure", choices=DISCLOSURES, default="explicit")
    parser.add_argument("--graph-json", type=Path)
    parser.add_argument("--graph-radius-m", type=float, default=1.0)
    parser.add_argument("--trajectory-h5", type=Path)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--model-mode", choices=["mock", "command", "http"], default="mock")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--protocol", default="openai_chat")
    parser.add_argument("--command", default="")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--image-detail", default="low")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = load_episodes(args.benchmark)
    _, source = select_episode(
        episodes, episode_index=args.episode_index, case_id=args.case_id
    )
    plan = build_grounded_plan(source)
    graph_context = load_graph_context(
        args.graph_json, plan=plan, radius_m=args.graph_radius_m
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "rule":
        outputs = []
        for disclosure in DISCLOSURES:
            generated = rule_instruction(plan, disclosure)
            outputs.append(
                apply_instruction(
                    source,
                    generated,
                    generation_mode="rule",
                    graph_context=graph_context,
                )
            )
        write_json(args.output_dir / "benchmark.json", outputs)
        write_json(args.output_dir / "grounded_plan.json", plan)
        if graph_context is not None:
            write_json(args.output_dir / "path_corridor_graph.json", graph_context)
    else:
        images: list[Path] = []
        frame_indices: list[int] = []
        if args.mode == "vlm":
            if args.trajectory_h5 is None:
                raise ValueError("VLM mode requires --trajectory-h5")
            images, frame_indices = extract_full_rollout_keyframes(
                args.trajectory_h5,
                args.output_dir / "keyframes",
                camera_name=args.camera_name,
            )
        generated, metrics = model_instruction(
            plan=plan,
            disclosure=args.disclosure,
            graph_context=graph_context,
            images=images,
            args=args,
        )
        if frame_indices:
            generated["evidence_frame_indices"] = frame_indices
        output = apply_instruction(
            source,
            generated,
            generation_mode=args.mode,
            graph_context=graph_context,
            model_metrics=metrics,
        )
        write_json(args.output_dir / "benchmark.json", [output])
        write_json(args.output_dir / "grounded_plan.json", plan)
        write_json(args.output_dir / "model_output.json", generated)
    print(args.output_dir / "benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
