"""Build and replay a public-only Module-2 decision dataset.

The extractor joins the exact candidate options and recorded M2 response in
``mllm_metrics.jsonl`` with the matching public semantic snapshot in
``debug/raw/step_boundaries.jsonl.gz``.  Evaluator-private observations and
task ground truth are never copied into a case.  Older recordings did not
persist the complete M2 request, so their compact graph/mission context is
reconstructed with the production projection helpers and marked accordingly.

The replay command calls only the MLLM endpoint.  It does not import ROS or run
the simulator, making it suitable for prompt/model/timeout comparisons.
"""

from __future__ import annotations

# ``evaluation/types.py`` would otherwise shadow the stdlib module when this
# file is invoked directly and argparse imports enum.  Match the other
# stand-alone evaluation utilities and remove only this script directory.
import sys as _sys

if __name__ == "__main__" and _sys.path:
    _direct_script_dir = _sys.path[0]
    if _direct_script_dir:
        _sys.path = [entry for entry in _sys.path if entry != _direct_script_dir]

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


CASE_SCHEMA_VERSION = "interactive_nav_m2_replay_case_v1"
DATASET_SCHEMA_VERSION = "interactive_nav_m2_replay_dataset_v1"
PREDICTION_SCHEMA_VERSION = "interactive_nav_m2_replay_prediction_v1"

# These names are forbidden at every depth of the model-facing request.  The
# list intentionally mirrors the restricted V3 target contract and adds the
# recorder's private/evaluator-only payload names.  Labels/references live
# outside ``request`` and are never sent to the model.
FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "gt_observations",
        "interaction_requirement",
        "oracle_plan",
        "required_interaction_ids",
        "scene_modifications",
        "selected_instance",
        "target_container_instance_id",
        "target_container_labels",
        "target_container_name",
        "target_container_source_object_name",
        "target_instance_id",
        "target_source_object_name",
    }
)

REQUEST_KEY_ORDER = (
    "schema_version",
    "instruction",
    "mission",
    "robot",
    "recent_decisions",
    "graph",
    "room_object_reasoning",
    "candidates",
)
REQUEST_KEYS = frozenset(REQUEST_KEY_ORDER)

ALLOWED_REASON_CODES = frozenset(
    {
        "TARGET_VISIBLE",
        "REVEAL_TARGET_CONTAINER",
        "UNLOCK_ROUTE",
        "EXPLORE_TARGET_ROOM",
        "INFORMATION_GAIN",
        "INTERACTION_COVERAGE",
        "RECOVERY_DIVERSIFICATION",
        "DISTANCE_TIEBREAK",
        "NO_SEMANTIC_PREFERENCE",
    }
)
ALLOWED_CONFIDENCE_CODES = frozenset({"low", "medium", "high"})


class DatasetError(ValueError):
    """A replay artifact violates the public-only dataset contract."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _install_runtime_import_paths() -> None:
    """Expose the two pure-Python semantic packages without sourcing ROS."""

    repository = _repo_root()
    roots = (
        repository
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_decision_py_pkg"
        / "scripts",
        repository
        / "Interactive-Nav-SG-nav"
        / "src"
        / "semantic_mllm_py_pkg"
        / "scripts",
    )
    for root in reversed(roots):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _runtime_symbols() -> dict[str, Any]:
    _install_runtime_import_paths()
    from semantic_decision_py_pkg.behavior_candidates import BehaviorCandidate
    from semantic_decision_py_pkg.model_policy import (
        ModelPolicyClient,
        ModelPolicyConfig,
        build_subgoal_selection_response_schema,
    )
    from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig

    return {
        "BehaviorCandidate": BehaviorCandidate,
        "ModelPolicyClient": ModelPolicyClient,
        "ModelPolicyConfig": ModelPolicyConfig,
        "build_response_schema": build_subgoal_selection_response_schema,
        "MLLMClient": MLLMClient,
        "MLLMClientConfig": MLLMClientConfig,
    }


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise DatasetError(f"{path}:{line_number}: expected JSON object")
                yield value
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc


def _nested_forbidden_paths(value: Any, prefix: str = "request") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if key.casefold() in FORBIDDEN_REQUEST_KEYS:
                found.append(path)
            found.extend(_nested_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_nested_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def audit_public_request(request: Mapping[str, Any]) -> None:
    """Reject private GT and malformed model-facing request fields."""

    extra = set(request) - REQUEST_KEYS
    missing = REQUEST_KEYS - set(request)
    if extra:
        raise DatasetError(f"request has unsupported keys: {sorted(extra)}")
    if missing:
        raise DatasetError(f"request is missing keys: {sorted(missing)}")
    forbidden = _nested_forbidden_paths(request)
    if forbidden:
        raise DatasetError(
            "request contains evaluator-private fields: " + ", ".join(forbidden)
        )
    if not str(request.get("instruction") or "").strip():
        raise DatasetError("request instruction is empty")
    candidates = request.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DatasetError("request candidates must be a non-empty array")
    candidate_ids = [str(item.get("id") or "") for item in candidates if isinstance(item, dict)]
    if len(candidate_ids) != len(candidates) or any(not value for value in candidate_ids):
        raise DatasetError("every request candidate must have a non-empty id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise DatasetError("request candidate ids must be unique")


def _parse_model_json(raw_text: Any) -> dict[str, Any] | None:
    if isinstance(raw_text, dict):
        return dict(raw_text)
    text = str(raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, dict) else None


def validate_response(
    response: Mapping[str, Any] | None,
    candidate_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, str]:
    """Validate the strict M2 response without importing the simulator."""

    if not isinstance(response, Mapping):
        return None, "response_not_object"
    if set(response) != {"ranked_ids", "reason", "confidence"}:
        return None, "response_keys_invalid"
    ranked = response.get("ranked_ids")
    if not isinstance(ranked, list) or not 1 <= len(ranked) <= 3:
        return None, "ranked_ids_length_invalid"
    normalized = [str(value or "") for value in ranked]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        return None, "ranked_ids_invalid_or_duplicate"
    allowed = set(candidate_ids)
    unknown = [value for value in normalized if value not in allowed]
    if unknown:
        return None, "ranked_ids_not_in_current_candidates:" + ",".join(unknown)
    reason = str(response.get("reason") or "")
    confidence = str(response.get("confidence") or "")
    if reason not in ALLOWED_REASON_CODES:
        return None, f"reason_invalid:{reason}"
    if confidence not in ALLOWED_CONFIDENCE_CODES:
        return None, f"confidence_invalid:{confidence}"
    return {
        "ranked_ids": normalized,
        "reason": reason,
        "confidence": confidence,
    }, ""


def _metric_request_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("episode_id") or ""),
        int(row.get("graph_revision", 0) or 0),
        int(row.get("candidate_sequence", 0) or 0),
    )


def _load_matching_snapshots(
    path: Path,
    wanted: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Read a recorder once and retain only exact M2 input revisions."""

    matches: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not path.is_file() or not wanted:
        return matches
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    boundary = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                semantic = boundary.get("semantic_candidates") or {}
                graph = boundary.get("unified_graph") or {}
                episode_id = str(
                    semantic.get("episode_id") or graph.get("episode_id") or ""
                )
                sequence = int(semantic.get("sequence", 0) or 0)
                revision = int(semantic.get("graph_revision", 0) or 0)
                key = (episode_id, revision, sequence)
                if key not in wanted or key in matches:
                    continue
                # Deliberately whitelist only public producer messages.  In
                # particular, never retain boundary["gt_observations"].
                matches[key] = {
                    "step_index": int(boundary.get("step_index", 0) or 0),
                    "semantic_candidates": dict(semantic),
                    "unified_graph": dict(graph),
                }
                if len(matches) == len(wanted):
                    break
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    return matches


def _request_from_recorded_public_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("public_request", "m2_public_request", "request_payload"):
        value = row.get(key)
        if isinstance(value, dict):
            request = {name: value.get(name) for name in REQUEST_KEY_ORDER}
            if all(name in value for name in REQUEST_KEY_ORDER):
                return request
    return None


def _reconstruct_request(
    row: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    symbols = _runtime_symbols()
    BehaviorCandidate = symbols["BehaviorCandidate"]
    ModelPolicyClient = symbols["ModelPolicyClient"]
    ModelPolicyConfig = symbols["ModelPolicyConfig"]

    semantic = dict(snapshot.get("semantic_candidates") or {})
    graph = dict(snapshot.get("unified_graph") or {})
    raw_candidates = []
    for item in semantic.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        try:
            raw_candidates.append(BehaviorCandidate(**item))
        except TypeError as exc:
            raise DatasetError(
                f"candidate {item.get('candidate_id', '<unknown>')} cannot be reconstructed: {exc}"
            ) from exc
    if not raw_candidates:
        raise DatasetError("matching semantic snapshot contains no raw candidates")

    client = ModelPolicyClient(
        ModelPolicyConfig(
            mode="disabled",
            selection_granularity=str(row.get("selection_granularity") or "candidate"),
        )
    )
    robot_context = {
        "robot_xy": list(semantic.get("robot_xy") or []),
        "exploration_context": dict(semantic.get("exploration_context") or {}),
        # Compact recordings do not persist the exact policy-owned history.
        # Keep it empty rather than infer or accidentally introduce future
        # execution evidence into this historical request.
        "decision_history": [],
        "group_history": [],
        "candidate_history": {},
        "entered_room_ids": [],
    }
    request = client.build_request(
        raw_candidates,
        dict(semantic.get("target_context") or {}),
        graph,
        robot_context=robot_context,
    )
    # mllm_metrics stores the options exactly as transmitted after curation;
    # use them instead of re-running a possibly newer curator implementation.
    request["candidates"] = [
        dict(item) for item in row.get("candidate_options") or [] if isinstance(item, dict)
    ]
    request["recent_decisions"] = []
    return {key: request.get(key) for key in REQUEST_KEY_ORDER}


def _stable_case_id(
    attempt_relative: str,
    row: Mapping[str, Any],
    ordinal: int,
) -> str:
    identity = "|".join(
        (
            attempt_relative,
            str(row.get("episode_id") or ""),
            str(row.get("graph_revision", 0) or 0),
            str(row.get("candidate_sequence", 0) or 0),
            str(ordinal),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    episode = str(row.get("episode_id") or "episode_unknown").replace("/", "_")
    return f"{episode}-g{int(row.get('graph_revision', 0) or 0)}-c{int(row.get('candidate_sequence', 0) or 0)}-{digest}"


def _attempt_relative_path(attempt_dir: Path, input_root: Path) -> str:
    try:
        return attempt_dir.relative_to(input_root).as_posix() or "."
    except ValueError:
        return attempt_dir.name


def _find_metrics_paths(input_root: Path) -> list[Path]:
    if input_root.is_file():
        if input_root.name != "mllm_metrics.jsonl":
            raise DatasetError("file input must be named mllm_metrics.jsonl")
        return [input_root]
    direct = input_root / "mllm_metrics.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(input_root.rglob("mllm_metrics.jsonl"))


def _episode_selected(attempt_dir: Path, episode_filters: set[str]) -> bool:
    if not episode_filters:
        return True
    names = {part for part in attempt_dir.parts}
    normalized = {value.removeprefix("episode_") for value in episode_filters}
    for name in names:
        if name in episode_filters:
            return True
        if name.startswith("episode_") and name.removeprefix("episode_") in normalized:
            return True
    return False


def extract_cases(
    input_root: Path,
    *,
    episode_filters: Iterable[str] = (),
    max_cases: int | None = None,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract M2 requests, returning cases and non-fatal diagnostics."""

    input_root = input_root.expanduser().resolve()
    filters = {str(value) for value in episode_filters if str(value)}
    metrics_paths = [
        path
        for path in _find_metrics_paths(input_root)
        if _episode_selected(path.parent, filters)
    ]
    if not metrics_paths:
        raise DatasetError(f"no mllm_metrics.jsonl found below {input_root}")

    cases: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_case_ids: set[str] = set()
    for metrics_path in metrics_paths:
        attempt_dir = metrics_path.parent
        rows = [row for row in _jsonl_rows(metrics_path) if row.get("role") == "subgoal_selection"]
        if not rows:
            continue
        wanted = {_metric_request_key(row) for row in rows}
        snapshots = _load_matching_snapshots(
            attempt_dir / "debug" / "raw" / "step_boundaries.jsonl.gz",
            wanted,
        )
        relative = _attempt_relative_path(attempt_dir, input_root)
        for ordinal, row in enumerate(rows):
            if max_cases is not None and len(cases) >= max(0, max_cases):
                return cases, warnings
            key = _metric_request_key(row)
            exact_request = _request_from_recorded_public_payload(row)
            reconstruction: dict[str, Any]
            try:
                if exact_request is not None:
                    request = exact_request
                    reconstruction = {
                        "mode": "recorded_public_request",
                        "missing_fields": [],
                    }
                    recorder_step = None
                else:
                    snapshot = snapshots.get(key)
                    if snapshot is None:
                        raise DatasetError(
                            "matching public snapshot missing for "
                            f"episode={key[0]} graph_revision={key[1]} candidate_sequence={key[2]}"
                        )
                    request = _reconstruct_request(row, snapshot)
                    reconstruction = {
                        "mode": "reconstructed_public_context",
                        "missing_fields": [
                            "recent_decisions",
                            "entered_room_ids",
                            "candidate_history",
                        ],
                    }
                    recorder_step = int(snapshot.get("step_index", 0) or 0)
                audit_public_request(request)
            except DatasetError as exc:
                message = f"{relative} row {ordinal + 1}: {exc}"
                if strict:
                    raise DatasetError(message) from exc
                warnings.append(message)
                continue

            candidate_ids = [str(item["id"]) for item in request["candidates"]]
            parsed = _parse_model_json(row.get("raw_text"))
            recorded_response, recorded_error = validate_response(parsed, candidate_ids)
            case_id = _stable_case_id(relative, row, ordinal)
            if case_id in seen_case_ids:
                raise DatasetError(f"duplicate generated case id: {case_id}")
            seen_case_ids.add(case_id)
            case = {
                "schema_version": CASE_SCHEMA_VERSION,
                "case_id": case_id,
                "source": {
                    "attempt": relative,
                    "episode_id": str(row.get("episode_id") or ""),
                    "graph_revision": int(row.get("graph_revision", 0) or 0),
                    "candidate_sequence": int(row.get("candidate_sequence", 0) or 0),
                    "recorder_step": recorder_step,
                    "recorded_model": str(row.get("model") or ""),
                    "recorded_timestamp": float(row.get("timestamp", 0.0) or 0.0),
                },
                "request": request,
                "reference": {
                    "kind": "recorded_policy_output",
                    "is_correctness_label": False,
                    "response": recorded_response,
                    "validation_error": recorded_error or str(row.get("error") or ""),
                },
                "annotation": {
                    "status": "unreviewed",
                    "acceptable_top1_ids": [],
                    "preferred_ranking": [],
                    "forbidden_ids": [],
                    "notes": "",
                },
                "reconstruction": reconstruction,
            }
            cases.append(case)
    return cases, warnings


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_dataset(
    output_dir: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    reconstruction_counts = Counter(
        str((case.get("reconstruction") or {}).get("mode") or "unknown") for case in cases
    )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": time.time(),
        "case_count": len(cases),
        "cases_file": cases_path.name,
        "cases_sha256": digest,
        "reconstruction_counts": dict(sorted(reconstruction_counts.items())),
        "warning_count": len(warnings),
        "warnings": list(warnings),
        "privacy_contract": {
            "model_input_is_public_only": True,
            "forbidden_request_keys": sorted(FORBIDDEN_REQUEST_KEYS),
            "recorded_response_is_not_correctness_label": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    annotation_rows = [
        {
            "case_id": str(case.get("case_id") or ""),
            "status": "unreviewed",
            "acceptable_top1_ids": [],
            "preferred_ranking": [],
            "forbidden_ids": [],
            "notes": "",
        }
        for case in cases
    ]
    _write_jsonl(output_dir / "annotations.template.jsonl", annotation_rows)
    return manifest


def load_cases(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "cases.jsonl"
    cases = list(_jsonl_rows(path))
    case_ids: set[str] = set()
    for line_number, case in enumerate(cases, start=1):
        if case.get("schema_version") != CASE_SCHEMA_VERSION:
            raise DatasetError(f"{path}:{line_number}: unsupported case schema")
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in case_ids:
            raise DatasetError(f"{path}:{line_number}: missing or duplicate case_id")
        case_ids.add(case_id)
        request = case.get("request")
        if not isinstance(request, dict):
            raise DatasetError(f"{path}:{line_number}: request must be an object")
        audit_public_request(request)
    return cases


def load_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for row in _jsonl_rows(path.expanduser().resolve()):
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise DatasetError(f"annotation without case_id in {path}")
        if case_id in annotations:
            raise DatasetError(f"duplicate annotation for {case_id}")
        annotations[case_id] = dict(row)
    return annotations


def _rank_of(candidate_id: str, ranking: Sequence[str]) -> int | None:
    try:
        return list(ranking).index(candidate_id) + 1
    except ValueError:
        return None


def score_prediction(
    case: Mapping[str, Any],
    response: Mapping[str, Any] | None,
    *,
    annotation_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = dict(case.get("request") or {})
    candidate_ids = [str(item.get("id") or "") for item in request.get("candidates") or []]
    validated, validation_error = validate_response(response, candidate_ids)
    annotation = dict(case.get("annotation") or {})
    if annotation_override is not None:
        annotation.update(dict(annotation_override))
    reference = dict((case.get("reference") or {}).get("response") or {})
    reference_ranking = [str(value) for value in reference.get("ranked_ids") or []]
    predicted_ranking = list((validated or {}).get("ranked_ids") or [])
    top1 = predicted_ranking[0] if predicted_ranking else ""
    reference_top1 = reference_ranking[0] if reference_ranking else ""
    acceptable = {
        str(value) for value in annotation.get("acceptable_top1_ids") or [] if str(value)
    }
    forbidden = {str(value) for value in annotation.get("forbidden_ids") or [] if str(value)}
    preferred = [str(value) for value in annotation.get("preferred_ranking") or [] if str(value)]
    preferred_rank = min(
        (
            rank
            for candidate_id in preferred
            if (rank := _rank_of(candidate_id, predicted_ranking)) is not None
        ),
        default=None,
    )
    overlap = (
        len(set(predicted_ranking[:3]).intersection(reference_ranking[:3]))
        / max(1, len(set(reference_ranking[:3])))
        if reference_ranking
        else None
    )
    return {
        "valid": validated is not None,
        "validation_error": validation_error,
        "response": validated,
        "top1": top1,
        "recorded_reference_available": bool(reference_ranking),
        "recorded_top1_agreement": (
            bool(top1 == reference_top1) if reference_top1 else None
        ),
        "recorded_top3_overlap": overlap,
        "annotation_status": str(annotation.get("status") or "unreviewed"),
        "annotated": bool(acceptable or preferred or forbidden),
        "acceptable_top1": bool(top1 in acceptable) if acceptable else None,
        "forbidden_top1": bool(top1 in forbidden) if forbidden else None,
        "preferred_reciprocal_rank": (
            1.0 / preferred_rank if preferred_rank is not None else 0.0
        )
        if preferred
        else None,
    }


def summarize_scores(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [dict(row.get("score") or {}) for row in predictions]

    def mean_optional(key: str) -> float | None:
        values = [float(row[key]) for row in scores if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    latencies = [
        float(row.get("latency_s", 0.0) or 0.0)
        for row in predictions
        if float(row.get("latency_s", 0.0) or 0.0) >= 0.0
    ]
    total_latencies = [
        float(row.get("total_latency_s", row.get("latency_s", 0.0)) or 0.0)
        for row in predictions
    ]
    errors = Counter(
        str(row.get("transport_error") or row.get("score", {}).get("validation_error") or "")
        for row in predictions
        if row.get("transport_error") or row.get("score", {}).get("validation_error")
    )
    return {
        "schema_version": "interactive_nav_m2_replay_summary_v1",
        "case_count": len(predictions),
        "valid_response_count": sum(bool(row.get("valid")) for row in scores),
        "valid_response_rate": mean_optional("valid"),
        "transport_error_count": sum(bool(row.get("transport_error")) for row in predictions),
        "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in predictions),
        "recorded_reference_case_count": sum(
            bool(row.get("recorded_reference_available")) for row in scores
        ),
        "recorded_top1_agreement": mean_optional("recorded_top1_agreement"),
        "recorded_top3_overlap": mean_optional("recorded_top3_overlap"),
        "annotated_case_count": sum(bool(row.get("annotated")) for row in scores),
        "acceptable_top1_case_count": sum(
            row.get("acceptable_top1") is not None for row in scores
        ),
        "acceptable_top1_accuracy": mean_optional("acceptable_top1"),
        "forbidden_top1_rate": mean_optional("forbidden_top1"),
        "preferred_mrr": mean_optional("preferred_reciprocal_rank"),
        "mean_latency_s": statistics.fmean(latencies) if latencies else None,
        "p95_latency_s": _percentile(latencies, 0.95),
        "mean_total_latency_s": statistics.fmean(total_latencies) if total_latencies else None,
        "p95_total_latency_s": _percentile(total_latencies, 0.95),
        "request_attempt_count": sum(len(row.get("attempts") or []) for row in predictions),
        "error_counts": dict(sorted(errors.items())),
        "metric_notes": {
            "recorded_agreement_is_not_accuracy": True,
            "correctness_metrics_require_annotations": True,
            "failed_predictions_count_as_incorrect_on_labeled_cases": True,
            "latency_s_sums_attempt_service_latency": True,
            "total_latency_s_includes_rate_limit_wait": True,
        },
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[position]


RequestFunction = Callable[[Mapping[str, Any], str], tuple[dict[str, Any] | None, dict[str, Any]]]


class SlidingWindowRateLimiter:
    """One thread-safe rolling request budget, shareable across model replays.

    Every attempt, including retries, must acquire a slot immediately before
    calling the transport.  The small boundary margin avoids rounding/scheduler
    jitter around the end of a 60-second window.  Zero disables throttling.
    """

    def __init__(
        self,
        requests_per_minute: int = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        boundary_margin_s: float = 0.05,
    ) -> None:
        if requests_per_minute < 0:
            raise DatasetError("requests_per_minute must be non-negative")
        self.requests_per_minute = int(requests_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._window_s = 60.0 + max(0.0, boundary_margin_s)
        self._lock = threading.Lock()
        self._recent: deque[float] = deque()
        self._admissions: list[float] = []

    @property
    def admission_timestamps(self) -> list[float]:
        with self._lock:
            return list(self._admissions)

    def acquire(self) -> float:
        started = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                while self._recent and now - self._recent[0] >= self._window_s:
                    self._recent.popleft()
                if not self.requests_per_minute or len(self._recent) < self.requests_per_minute:
                    self._recent.append(now)
                    self._admissions.append(now)
                    return max(0.0, now - started)
                wait_s = self._recent[0] + self._window_s - now
            # Do not hold the lock while waiting: simultaneous model runs share
            # this limiter, and completed workers must remain observable.
            self._sleep(max(wait_s, 0.001))


def _is_timeout_error(error: Any) -> bool:
    normalized = " ".join(str(error or "").casefold().split())
    return any(
        marker in normalized
        for marker in (
            "timed out",
            "timeout",
            "deadline exceeded",
            "http 408",
            "http 504",
        )
    )


def replay_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    request_function: RequestFunction,
    annotations: Mapping[str, Mapping[str, Any]] | None = None,
    max_retries: int = 0,
    prompt_override: str = "",
    concurrency: int = 1,
    requests_per_minute: int = 0,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    on_prediction: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Replay in parallel, reporting completions while returning input order.

    ``request_function`` must support concurrent calls when concurrency > 1.
    ``on_prediction`` is called serially by the coordinator, not worker threads.
    Share one ``rate_limiter`` between invocations to impose an aggregate budget.
    """
    if concurrency < 1:
        raise DatasetError("concurrency must be at least 1")
    limiter = rate_limiter or SlidingWindowRateLimiter(requests_per_minute)
    annotations = annotations or {}

    def replay_case(case: Mapping[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        case_id = str(case.get("case_id") or "")
        request = dict(case.get("request") or {})
        if prompt_override:
            request["instruction"] = prompt_override
        audit_public_request(request)
        response: dict[str, Any] | None = None
        request_metrics: dict[str, Any] = {}
        attempt_metrics: list[dict[str, Any]] = []
        retry_count = 0
        for attempt in range(max(0, int(max_retries)) + 1):
            rate_limit_wait_s = limiter.acquire()
            attempt_started = time.monotonic()
            attempt_started_at = time.time()
            try:
                response, request_metrics = request_function(request, case_id)
            except Exception as exc:
                # Preserve failed cases in the accuracy denominator, including
                # an injected transport that raises instead of returning errors.
                response = None
                request_metrics = {"error": f"{type(exc).__name__}: {exc}"}
            request_metrics = dict(request_metrics)
            request_metrics.setdefault("latency_s", time.monotonic() - attempt_started)
            error = str(request_metrics.get("error") or "")
            attempt_metrics.append(
                {
                    "attempt": attempt + 1,
                    "started_at": attempt_started_at,
                    "started_monotonic": attempt_started,
                    "rate_limit_wait_s": rate_limit_wait_s,
                    "error": error,
                    "latency_s": float(request_metrics.get("latency_s", 0.0) or 0.0),
                    "prompt_tokens": int(request_metrics.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        request_metrics.get("completion_tokens", 0) or 0
                    ),
                    "reasoning_tokens": int(request_metrics.get("reasoning_tokens", 0) or 0),
                    "total_tokens": int(request_metrics.get("total_tokens", 0) or 0),
                }
            )
            if response is not None and not error:
                break
            if attempt < max(0, int(max_retries)) and _is_timeout_error(error):
                retry_count += 1
                continue
            break
        score = score_prediction(
            case,
            response if not request_metrics.get("error") else None,
            annotation_override=annotations.get(case_id),
        )
        return {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "case_id": case_id,
            "response": response,
            "valid": bool(score["valid"]),
            "transport_error": str(request_metrics.get("error") or ""),
            "retry_count": retry_count,
            "attempts": attempt_metrics,
            "latency_s": sum(item["latency_s"] for item in attempt_metrics),
            "total_latency_s": time.monotonic() - started,
            "rate_limit_wait_s": sum(item["rate_limit_wait_s"] for item in attempt_metrics),
            "prompt_tokens": sum(item["prompt_tokens"] for item in attempt_metrics),
            "completion_tokens": sum(item["completion_tokens"] for item in attempt_metrics),
            "reasoning_tokens": sum(item["reasoning_tokens"] for item in attempt_metrics),
            "total_tokens": sum(item["total_tokens"] for item in attempt_metrics),
            "score": score,
        }

    if concurrency == 1:
        sequential = []
        for case in cases:
            prediction = replay_case(case)
            sequential.append(prediction)
            if on_prediction is not None:
                on_prediction(prediction)
        return sequential

    predictions: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="m2-replay") as pool:
        futures = {pool.submit(replay_case, case): index for index, case in enumerate(cases)}
        for future in as_completed(futures):
            prediction = future.result()
            predictions[futures[future]] = prediction
            if on_prediction is not None:
                on_prediction(prediction)
    return [prediction for prediction in predictions if prediction is not None]


def build_mllm_request_function(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    metrics_path: str | None = None,
) -> RequestFunction:
    symbols = _runtime_symbols()
    MLLMClient = symbols["MLLMClient"]
    MLLMClientConfig = symbols["MLLMClientConfig"]
    build_response_schema = symbols["build_response_schema"]
    # MLLMClient keeps request configs local (dataclasses.replace), creates a
    # fresh transport per request and guards metrics with a lock + flock.  One
    # shared instance preserves that log lock while allowing parallel requests.
    client = MLLMClient(
        MLLMClientConfig(
            mode=args.mode,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            model=args.model,
            protocol=args.protocol,
            command=args.command,
            timeout_s=args.timeout_s,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            metrics_path=(
                str(output_dir / "transport_metrics.jsonl")
                if metrics_path is None
                else metrics_path
            ),
        )
    )

    def request_function(
        public_request: Mapping[str, Any], case_id: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        payload = dict(public_request)
        response = client.request_json(
            role="subgoal_selection",
            instruction=str(payload["instruction"]),
            context={
                "mission": payload["mission"],
                "robot": payload["robot"],
                "graph": payload["graph"],
                "room_object_reasoning": payload["room_object_reasoning"],
                "candidates": payload["candidates"],
                "recent_decisions": payload["recent_decisions"],
            },
            response_schema=build_response_schema(payload),
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            metrics_context={"case_id": case_id, "dataset_role": "m2_offline_replay"},
        )
        return response.payload, response.metrics()

    return request_function


def _extract_command(args: argparse.Namespace) -> int:
    cases, warnings = extract_cases(
        Path(args.input_root),
        episode_filters=args.episode,
        max_cases=args.max_cases,
        strict=args.strict,
    )
    manifest = write_dataset(Path(args.output_dir), cases, warnings=warnings)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if cases else 2


def _replay_command(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.dataset))
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if str(case.get("case_id") or "") in selected]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise DatasetError("no replay cases selected")
    annotations = load_annotations(Path(args.annotations) if args.annotations else None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_override = (
        Path(args.prompt_file).expanduser().read_text(encoding="utf-8").strip()
        if args.prompt_file
        else ""
    )
    request_function = build_mllm_request_function(args, output_dir)
    completed: list[dict[str, Any]] = []
    replay_started = time.monotonic()
    # Completion-order journal remains readable if the process is interrupted.
    # The final predictions.jsonl below retains deterministic dataset order.
    with (output_dir / "predictions.inprogress.jsonl").open("w", encoding="utf-8") as stream:
        def report_prediction(prediction: dict[str, Any]) -> None:
            stream.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            completed.append(prediction)
            progress = summarize_scores(completed)
            print(json.dumps({
                "event": "m2_replay_progress",
                "completed": len(completed),
                "total": len(cases),
                "valid_response_rate": progress["valid_response_rate"],
                "acceptable_top1_accuracy": progress["acceptable_top1_accuracy"],
                "elapsed_s": round(time.monotonic() - replay_started, 3),
            }, ensure_ascii=False), file=sys.stderr, flush=True)

        predictions = replay_cases(
            cases,
            request_function=request_function,
            annotations=annotations,
            max_retries=args.max_retries,
            prompt_override=prompt_override,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            on_prediction=report_prediction,
        )
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    summary = summarize_scores(predictions)
    summary.update(
        {
            "created_at": time.time(),
            "model": args.model,
            "endpoint": args.endpoint,
            "protocol": args.protocol,
            "timeout_s": args.timeout_s,
            "max_retries": args.max_retries,
            "prompt_override": bool(prompt_override),
            "concurrency": args.concurrency,
            "requests_per_minute": args.requests_per_minute,
            "wall_time_s": time.monotonic() - replay_started,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["valid_response_count"] == summary["case_count"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and replay public-only InteractiveNav Module-2 decisions."
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    extract = subparsers.add_parser(
        "extract", help="join M2 metrics with public recorder snapshots"
    )
    extract.add_argument("--input-root", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument(
        "--episode",
        action="append",
        default=[],
        help="episode directory/name filter; may be repeated",
    )
    extract.add_argument("--max-cases", type=int)
    extract.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping requests without a matching public snapshot",
    )
    extract.set_defaults(handler=_extract_command)

    replay = subparsers.add_parser("replay", help="run M2 only, without ROS/simulation")
    replay.add_argument("--dataset", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--annotations")
    replay.add_argument("--case-id", action="append", default=[])
    replay.add_argument("--limit", type=int)
    replay.add_argument("--prompt-file")
    replay.add_argument("--mode", choices=("http", "command"), default="http")
    replay.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    replay.add_argument("--protocol", default="openai_chat")
    replay.add_argument("--model", default="qwen3.6-35b-a3b")
    replay.add_argument("--api-key-env", default="OPENAI_API_KEY")
    replay.add_argument("--command", default="")
    replay.add_argument("--timeout-s", type=float, default=30.0)
    replay.add_argument("--max-retries", type=int, default=1)
    replay.add_argument("--concurrency", type=int, default=1)
    replay.add_argument(
        "--requests-per-minute", type=int, default=0,
        help="aggregate rolling 60-second attempt budget, including retries; 0 is unlimited",
    )
    replay.add_argument("--max-tokens", type=int, default=1536)
    replay.add_argument("--temperature", type=float, default=0.0)
    replay.add_argument("--reasoning-effort", default="off")
    replay.set_defaults(handler=_replay_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except DatasetError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
