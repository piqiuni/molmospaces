"""Read-only summary for a parallel InteractiveNav V3 evaluation round.

The utility intentionally consumes completed artifacts instead of importing the
simulator or ROS.  It is therefore safe to run while another round is being
inspected and also works with older recordings that do not expose every RGB
alignment counter yet.
"""

from __future__ import annotations

# When this file is executed by path, Python prepends this ``evaluation``
# directory to sys.path.  Its legacy ``types.py`` would then shadow the stdlib
# module while argparse imports enum.  The report is self-contained, so remove
# only that direct-script entry before importing the standard library.
import sys as _sys

if __name__ == "__main__" and _sys.path:
    _direct_script_dir = _sys.path[0]
    if _direct_script_dir:
        _sys.path = [entry for entry in _sys.path if entry != _direct_script_dir]

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "interactive_nav_v3_round_summary_v5"
PAPER_METRIC_SCHEMA_VERSION = "interactive_nav_v3_paper_metrics_v1"
_WORKER_INDEX_RE = re.compile(r"^worker[_-]?(?P<index>\d+)")
_PRE_SCORE_MARKERS = {
    "pre_score_guard",
    "pre-score-guard",
    "prescore_guard",
}
_PRE_SCORE_CONTEXT_KEYS = {
    "decision_source",
    "fallback_source",
    "guard_source",
    "policy_source",
    "reason_code",
    "role",
    "selection_source",
    "source",
}
_BATCH_MANIFEST_NAME = "batch_manifest.json"
_BATCH_TASK_SUMMARY_NAME = "batch_task_summary.json"
_ATTEMPT_DIR_RE = re.compile(r"^attempt_(?P<index>\d+)$")
_M1_METRIC_ROLE = "attribute_inference"

# These public failure tokens are emitted before an evaluator-owned force skill
# can run.  In the restricted V3 evaluator they are therefore the public trace
# projection of a formally ``invalid`` interaction attempt.  Do not add normal
# execution/postcondition failures here: a required/extra valid interaction may
# fail physically and must remain distinguishable from an invalid request.
_INVALID_PRECONDITION_FAILURE_REASONS = frozenset(
    {
        "capability_unavailable",
        "interaction_not_visible",
        "interaction_pose_invalid",
        "interaction_pose_poll_exhausted",
        "interaction_too_far",
        "interaction_unsupported",
        "non_articulated",
        "non_articulated_blocked_portal",
        "non_articulated_closed_portal",
        "unknown_instance_id",
        "unresolved_drawer_scan_target",
        "unsupported_action",
    }
)
_FAILED_INTERACTION_STATUSES = frozenset(
    {
        "aborted",
        "error",
        "failed",
        "invalid",
        "rejected",
    }
)


@dataclass(frozen=True)
class _EpisodeArtifacts:
    """The evaluator evidence belonging to one logical round episode.

    Batch execution has a three-level layout (manifest -> task -> attempt),
    while the original single-worker layout places the evaluator result directly
    below a worker directory.  Keeping that normalization here gives callers a
    single ``summarise_round`` interface and prevents a stale retry from being
    reported as a second episode.
    """

    worker: str
    task_dir: Path
    attempt_dir: Path | None
    result_path: Path | None
    task_summary: dict[str, Any]
    manifest_plan: dict[str, Any] | None
    planned: bool


def _read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        warnings.append(f"Unable to read JSON {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        warnings.append(f"Expected a JSON object in {path}")
        return {}
    return payload


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        warnings.append(f"Unable to read JSONL {path}: {exc}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"Ignoring malformed JSONL row {path}:{line_number}: {exc}")
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            warnings.append(f"Ignoring non-object JSONL row {path}:{line_number}")
    return rows


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _optional_int(value: Any) -> int | None:
    number = _finite_number(value)
    return int(number) if number is not None else None


def _optional_bool(value: Any) -> bool | None:
    """Return a persisted Boolean without treating a missing value as false."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _json_object(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _paper_spl_from_saved_paths(
    nav_success: bool | None,
    reference_length_m: float | int | None,
    executed_length_m: float | int | None,
) -> float | None:
    """Recompute paper SPL from its saved NavToObj inputs.

    Older V3 rows stored an interaction-conditioned SPL value.  A post-hoc
    report must not resurrect that definition merely because a `spl` scalar is
    present in the JSON.
    """

    if nav_success is None or reference_length_m is None or executed_length_m is None:
        return None
    reference = max(0.0, float(reference_length_m))
    executed = max(0.0, float(executed_length_m))
    if not nav_success:
        return 0.0
    return reference / max(reference, executed, 1e-9)


def _worker_dir(round_root: Path, result_path: Path) -> Path:
    for parent in result_path.parents:
        if parent.name.startswith("worker"):
            return parent
        if parent == round_root:
            break
    try:
        relative = result_path.relative_to(round_root)
    except ValueError:
        return result_path.parent
    return round_root / relative.parts[0] if len(relative.parts) > 1 else round_root


def _worker_sort_key(name: str) -> tuple[int, str]:
    match = _WORKER_INDEX_RE.match(name)
    return (int(match.group("index")), name) if match else (10**9, name)


def discover_episode_results(round_root: Path) -> list[Path]:
    """Find all evaluator episode results below *round_root* deterministically."""

    if not round_root.is_dir():
        raise NotADirectoryError(f"Round root is not a directory: {round_root}")
    paths = [path for path in round_root.rglob("episode_result.json") if path.is_file()]

    def sort_key(path: Path) -> tuple[tuple[int, str], int, str]:
        worker = _worker_dir(round_root, path).name
        episode_match = re.search(r"(?:^|[_-])ep(?:isode)?[_-]?(\d+)", worker, re.IGNORECASE)
        episode_hint = int(episode_match.group(1)) if episode_match else 10**9
        return _worker_sort_key(worker), episode_hint, str(path)

    return sorted(paths, key=sort_key)


def _path_from_payload(value: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _attempt_sort_key(path: Path) -> tuple[int, str]:
    match = _ATTEMPT_DIR_RE.match(path.name)
    return (int(match.group("index")), path.name) if match else (-1, path.name)


def _latest_attempt_dir(task_dir: Path, task_summary: dict[str, Any]) -> Path | None:
    """Return the latest persisted batch attempt, preferring numeric retries."""

    candidates = [
        path
        for path in task_dir.glob("attempt_*")
        if path.is_dir() and _ATTEMPT_DIR_RE.match(path.name)
    ]
    summary_attempt = _path_from_payload(task_summary.get("attempt_dir"), base_dir=task_dir)
    if summary_attempt is not None and summary_attempt.is_dir():
        candidates.append(summary_attempt)
    if not candidates:
        return None
    return max(set(candidates), key=_attempt_sort_key)


def _result_path_for_attempt(
    attempt_dir: Path | None,
    task_summary: dict[str, Any],
    warnings: list[str],
    *,
    planned_episode_index: int | None,
) -> Path | None:
    """Resolve the planned evaluator result without double-counting retries.

    A task can retain stale files after a failed retry.  A batch result is only
    valid when exactly one candidate under the selected attempt reports the
    manifest's episode index; ambiguity is evidence of an incomplete task, not
    a reason to choose a convenient file by timestamp.
    """

    summary_result = _path_from_payload(
        task_summary.get("episode_result_path"),
        base_dir=attempt_dir or Path.cwd(),
    )
    candidates: list[Path] = []
    if (
        summary_result is not None
        and summary_result.is_file()
        and (attempt_dir is None or attempt_dir in summary_result.parents)
    ):
        candidates.append(summary_result)
    if attempt_dir is not None:
        candidates.extend(
            path for path in attempt_dir.rglob("episode_result.json") if path.is_file()
        )
    candidates = sorted(set(candidates), key=str)
    if not candidates:
        return None

    if planned_episode_index is None:
        if len(candidates) == 1:
            return candidates[0]
        warnings.append(
            f"Ambiguous episode results without a planned index below {attempt_dir}; ignoring"
        )
        return None

    matching: list[Path] = []
    for candidate in candidates:
        payload = _result_payload(_read_json(candidate, warnings))
        if _optional_int(payload.get("episode_index")) == planned_episode_index:
            matching.append(candidate)
    if len(matching) == 1:
        return matching[0]
    if not matching:
        warnings.append(
            f"No result for planned episode {planned_episode_index} below {attempt_dir}"
        )
    else:
        warnings.append(
            f"Ambiguous results for planned episode {planned_episode_index} below {attempt_dir}"
        )
    return None


def _batch_artifacts(round_root: Path, warnings: list[str]) -> list[_EpisodeArtifacts] | None:
    """Normalize a ROS batch manifest into one artifact record per planned task."""

    manifest_path = round_root / _BATCH_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path, warnings)
    raw_plans = manifest.get("plans")
    if not isinstance(raw_plans, list):
        warnings.append(f"Batch manifest has no plans list: {manifest_path}")
        return None

    artifacts: list[_EpisodeArtifacts] = []
    for ordinal, raw_plan in enumerate(raw_plans):
        if not isinstance(raw_plan, dict):
            warnings.append(f"Ignoring non-object batch plan {manifest_path}:{ordinal}")
            continue
        plan = dict(raw_plan)
        episode_index = _optional_int(plan.get("episode_index"))
        task_dir = _path_from_payload(plan.get("output_dir"), base_dir=round_root)
        if task_dir is None:
            task_dir = round_root / (
                f"episode_{episode_index:04d}" if episode_index is not None else f"episode_{ordinal:04d}"
            )
        summary_path = task_dir / _BATCH_TASK_SUMMARY_NAME
        task_summary = _read_json(summary_path, warnings) if summary_path.is_file() else {}
        attempt_dir = _latest_attempt_dir(task_dir, task_summary)
        result_path = _result_path_for_attempt(
            attempt_dir,
            task_summary,
            warnings,
            planned_episode_index=episode_index,
        )
        worker_id = task_summary.get("worker_id", plan.get("worker_id"))
        worker = f"worker_{worker_id}" if _optional_int(worker_id) is not None else task_dir.name
        artifacts.append(
            _EpisodeArtifacts(
                worker=worker,
                task_dir=task_dir,
                attempt_dir=attempt_dir,
                result_path=result_path,
                task_summary=task_summary,
                manifest_plan=plan,
                planned=True,
            )
        )
    return artifacts


def _legacy_artifacts(round_root: Path) -> list[_EpisodeArtifacts]:
    """Adapt pre-batch/worker layouts to the round summary's internal seam."""

    artifacts: list[_EpisodeArtifacts] = []
    for result_path in discover_episode_results(round_root):
        worker_dir = _worker_dir(round_root, result_path)
        attempt_dir = next(
            (
                parent
                for parent in result_path.parents
                if _ATTEMPT_DIR_RE.match(parent.name)
            ),
            None,
        )
        artifacts.append(
            _EpisodeArtifacts(
                worker=worker_dir.name,
                task_dir=worker_dir,
                attempt_dir=attempt_dir,
                result_path=result_path,
                task_summary={},
                manifest_plan=None,
                planned=False,
            )
        )
    return artifacts


def _result_payload(document: dict[str, Any]) -> dict[str, Any]:
    nested = document.get("result")
    return nested if isinstance(nested, dict) else document


def _duplicate_interactions(attempts: Any) -> int:
    if not isinstance(attempts, list):
        return 0
    seen: set[tuple[str, str]] = set()
    duplicate_count = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        instance_id = str(attempt.get("instance_id") or "")
        operation = str(attempt.get("operation") or "")
        request_id = str(attempt.get("request_id") or "")
        # A request id is a useful fallback for incomplete/failed interaction
        # records, but strip the decision prefix so retries still compare equal.
        fallback = request_id.split(":", 1)[1] if ":" in request_id else request_id
        key = (instance_id or fallback, operation)
        if not any(key):
            continue
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return duplicate_count


def _coverage_payload(
    result_path: Path | None,
    worker_dir: Path,
    debug_summary: dict[str, Any],
    warnings: list[str],
) -> tuple[float | int | None, str | None, Path | None]:
    if result_path is not None:
        topdown_path = result_path.with_name("episode_topdown.json")
        if topdown_path.is_file():
            topdown = _read_json(topdown_path, warnings)
            coverage = topdown.get("coverage")
            if isinstance(coverage, dict):
                ratio = _finite_number(coverage.get("exploration_coverage_ratio"))
                if ratio is not None:
                    return ratio, str(coverage.get("source") or "episode_topdown"), topdown_path

    coverage_path = worker_dir / "debug" / "exploration_coverage.json"
    if coverage_path.is_file():
        coverage = _read_json(coverage_path, warnings)
        ratio = _finite_number(coverage.get("exploration_coverage_ratio"))
        if ratio is not None:
            return ratio, "debug_exploration_coverage", coverage_path

    ratio = _finite_number(debug_summary.get("exploration_coverage_ratio"))
    return ratio, "debug_summary" if ratio is not None else None, None


def _contains_pre_score_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if "pre_score_guard" in normalized_key:
                if isinstance(child, bool):
                    if child:
                        return True
                elif child not in (None, "", 0, [], {}):
                    return True
            if normalized_key in _PRE_SCORE_CONTEXT_KEYS and isinstance(child, str):
                normalized_child = child.strip().lower().replace(" ", "_")
                if (
                    normalized_child in _PRE_SCORE_MARKERS
                    or "pre_score_guard" in normalized_child.replace("-", "_")
                ):
                    return True
            # Do not scan arbitrary text leaves: prompts may describe the guard
            # without the guard actually having been applied.
            if isinstance(child, (dict, list)) and _contains_pre_score_marker(child):
                return True
    if isinstance(value, list):
        return any(_contains_pre_score_marker(child) for child in value)
    return False


def _mllm_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selection_rows = [row for row in rows if str(row.get("role") or "") == "subgoal_selection"]
    top_candidates: list[str] = []
    for row in selection_rows:
        candidate_ids = row.get("candidate_ids")
        if isinstance(candidate_ids, list) and candidate_ids:
            candidate_id = str(candidate_ids[0] or "")
        else:
            candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            top_candidates.append(candidate_id)

    counts = Counter(top_candidates)
    repeated_count = sum(count - 1 for count in counts.values() if count > 1)
    consecutive_repeat_count = sum(
        current == previous for previous, current in zip(top_candidates, top_candidates[1:])
    )
    repeated = [
        {"candidate_id": candidate_id, "selection_count": count}
        for candidate_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ]
    latencies = [
        float(value)
        for row in rows
        if (value := _finite_number(row.get("latency_s"))) is not None
    ]
    error_count = sum(bool(str(row.get("error") or "")) for row in rows)
    m1_rows = [row for row in rows if str(row.get("role") or "") == _M1_METRIC_ROLE]
    m1_errors = [str(row.get("error") or "").strip() for row in m1_rows]
    m1_errors = [error for error in m1_errors if error]
    return {
        "call_count": len(rows),
        "successful_call_count": len(rows) - error_count,
        "error_count": error_count,
        "m1_call_count": len(m1_rows),
        "m1_successful_call_count": len(m1_rows) - len(m1_errors),
        "m1_error_count": len(m1_errors),
        "m1_error_reason_counts": dict(sorted(Counter(m1_errors).items())),
        "pre_score_guard_count": sum(_contains_pre_score_marker(row) for row in rows),
        "subgoal_selection_call_count": len(selection_rows),
        "selected_candidate_count": len(top_candidates),
        "unique_selected_candidate_count": len(counts),
        "repeated_selected_candidate_count": repeated_count,
        "consecutive_repeat_count": consecutive_repeat_count,
        "max_candidate_selection_count": max(counts.values(), default=0),
        "most_repeated_candidates": repeated[:5],
        "latency_sum_s": sum(latencies),
    }


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
    return next((path for path in candidates if path.is_file()), None)


def _completion_state(
    artifacts: _EpisodeArtifacts,
    document: dict[str, Any],
    result: dict[str, Any],
) -> bool | None:
    """Use batch wrapper completion when available, otherwise infer safely."""

    explicit = _optional_bool(artifacts.task_summary.get("completed"))
    if artifacts.planned and explicit is None:
        # A batch task is complete only after its wrapper writes the final
        # validation record.  An evaluator JSON alone may have been flushed
        # immediately before a killed worker or a failed artifact check.
        return False
    if explicit is not None:
        # The wrapper's Boolean is necessary but not sufficient evidence after
        # files have been copied or partially cleaned: a completed task must
        # still have one evaluator result in the selected latest attempt.
        document_status = document.get("status")
        result_status = result.get("status", document_status)
        return bool(
            explicit
            and artifacts.result_path is not None
            and document_status in (None, "complete")
            and result_status in (None, "complete")
        )
    if not result:
        return False if artifacts.planned else None
    document_status = document.get("status")
    result_status = result.get("status", document_status)
    return bool(
        document_status in (None, "complete") and result_status in (None, "complete")
    )


def _interaction_diagnostics(attempts: Any) -> dict[str, Any]:
    """Summarise public invalid, failed, and unknown interaction outcomes.

    ``invalid_reason_counts`` is intentionally tied to the evaluator's formal
    invalid-attempt metric.  New restricted V3 traces report some invalid
    requests as ``FAILED`` while placing the public reason in
    ``failure_reason``; keeping those reasons only under a generic failure
    counter would leave the formal invalid total unexplained.  Conversely,
    physical force/postcondition failures stay in ``failed_reason_counts`` so
    this diagnostic never relabels a valid request as invalid.
    """

    if not isinstance(attempts, list):
        return {
            "invalid_reason_counts": {},
            "failed_reason_counts": {},
            "failed_interaction_attempt_count": 0,
            "unknown_reason_counts": {},
            "unknown_attempt_count": 0,
        }
    invalid_reasons: Counter[str] = Counter()
    failed_reasons: Counter[str] = Counter()
    unknown_reasons: Counter[str] = Counter()
    failed_interaction_attempt_count = 0
    unknown_attempt_count = 0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        status = str(attempt.get("status") or attempt.get("result_status") or "").strip()
        result_status = str(attempt.get("result_status") or "").strip()
        classification = str(attempt.get("classification") or "").strip()
        status_token = status.casefold()
        result_status_token = result_status.casefold()
        is_explicitly_invalid = (
            status_token == "invalid"
            or result_status_token == "invalid"
            or classification.casefold() == "invalid"
        )
        reason = str(
            attempt.get("reason") or attempt.get("failure_reason") or ""
        ).strip()
        normalized_reason = reason.casefold().replace("-", "_").replace(" ", "_")
        is_invalid_precondition_failure = (
            not is_explicitly_invalid
            and normalized_reason in _INVALID_PRECONDITION_FAILURE_REASONS
            and (
                status_token in _FAILED_INTERACTION_STATUSES
                or result_status_token in _FAILED_INTERACTION_STATUSES
            )
        )
        is_invalid = is_explicitly_invalid or is_invalid_precondition_failure
        is_unknown = (
            normalized_reason.startswith("unknown")
            or status_token == "unknown"
            or result_status_token == "unknown"
        )
        if is_invalid:
            invalid_reasons[reason or "unspecified"] += 1
        elif status_token in _FAILED_INTERACTION_STATUSES or result_status_token in _FAILED_INTERACTION_STATUSES:
            failed_reasons[reason or "unspecified"] += 1
            failed_interaction_attempt_count += 1
        if is_unknown:
            unknown_reasons[reason or "unknown_status"] += 1
            unknown_attempt_count += 1
    return {
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "failed_reason_counts": dict(sorted(failed_reasons.items())),
        "failed_interaction_attempt_count": failed_interaction_attempt_count,
        "unknown_reason_counts": dict(sorted(unknown_reasons.items())),
        "unknown_attempt_count": unknown_attempt_count,
    }


def _summarise_artifact(
    round_root: Path,
    artifacts: _EpisodeArtifacts,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    result_path = artifacts.result_path
    document = _read_json(result_path, warnings) if result_path is not None else {}
    result = _result_payload(document)
    task_summary = artifacts.task_summary
    plan = artifacts.manifest_plan or {}
    artifact_root = artifacts.attempt_dir or artifacts.task_dir
    debug_candidates = (artifact_root / "debug" / "summary.json",)
    mllm_candidates = (artifact_root / "mllm_metrics.jsonl",)
    if artifacts.attempt_dir is None:
        # Only legacy layouts use the task/worker directory as the artifact
        # root.  In a batch, falling back here would silently attribute an old
        # retry's MLLM/debug evidence to the latest attempt.
        debug_candidates = (artifacts.task_dir / "debug" / "summary.json",)
        mllm_candidates = (artifacts.task_dir / "mllm_metrics.jsonl",)
    debug_summary_path = _first_existing_path(debug_candidates)
    debug_summary = _read_json(debug_summary_path, warnings) if debug_summary_path else {}
    mllm_metrics_path = _first_existing_path(mllm_candidates)
    mllm_rows = _read_jsonl(mllm_metrics_path, warnings) if mllm_metrics_path else []
    coverage_ratio, coverage_source, coverage_path = _coverage_payload(
        result_path, artifact_root, debug_summary, warnings
    )

    actual_path = _finite_number(
        _first_present(result.get("navigation_path_length_m"), task_summary.get("navigation_path_length_m"))
    )
    reference_path = _finite_number(
        _first_present(result.get("reference_path_length_m"), task_summary.get("reference_path_length_m"))
    )
    attempts = result.get("interaction_attempts")
    attempt_count = len(attempts) if isinstance(attempts, list) else 0
    extra_count = _optional_int(
        _first_present(result.get("extra_interaction_action_count"), task_summary.get("extra_interaction_action_count"))
    ) or 0
    invalid_count = _optional_int(
        _first_present(result.get("invalid_interaction_action_count"), task_summary.get("invalid_interaction_action_count"))
    ) or 0
    raw_early_stop = _first_present(result.get("early_stop"), task_summary.get("early_stop"))
    raw_early_stop = raw_early_stop if isinstance(raw_early_stop, dict) else {}
    completed = _completion_state(artifacts, document, result)
    scoring_eligible = _optional_bool(result.get("scoring_eligible"))
    if artifacts.planned and completed is False:
        # The batch wrapper's evidence contract is stricter than an evaluator
        # JSON that happened to be written before the wrapper failed (for
        # example a missing required video or non-zero runner exit).  Retain
        # its payload for diagnosis but never treat it as a formal score.
        scoring_eligible = False
    elif scoring_eligible is None:
        # The evaluator's historical contract treats an omitted field as
        # eligible.  Preserve that contract for old artifacts, while all new
        # records write the Boolean explicitly.
        scoring_eligible = bool(_completion_state(artifacts, document, result) is not False)
    raw_requirement = _first_present(
        result.get("interaction_requirement"), task_summary.get("interaction_requirement")
    )
    interaction_requirement = (
        str(raw_requirement) if raw_requirement not in (None, "") else None
    )
    raw_paper_schema_version = result.get("paper_metric_schema_version")
    paper_metric_schema_version = (
        str(raw_paper_schema_version)
        if raw_paper_schema_version not in (None, "")
        else None
    )
    paper_metric_config = _json_object(result.get("paper_metric_config"))
    paper_nav_success = _optional_bool(result.get("nav_success"))
    paper_spl = _paper_spl_from_saved_paths(
        paper_nav_success,
        reference_path,
        actual_path,
    )

    expected_step_sync_count = _optional_int(
        _first_present(result.get("step_count"), task_summary.get("step_count"))
    )
    step_sync_count = _optional_int(debug_summary.get("step_sync_count"))
    video_frame_count = _optional_int(debug_summary.get("first_person_video_frame_count"))
    step_sync_capture_every = _optional_int(debug_summary.get("step_sync_capture_every"))
    # Recordings created before step-sampled six-panel video did not expose a
    # sampling interval or capture counter.  They captured every marker, so
    # retain their meaning by treating the interval as one and the video count
    # as the capture count.
    if step_sync_capture_every is None:
        step_sync_capture_every = 1
    elif step_sync_capture_every < 1:
        warnings.append(
            "Invalid step_sync_capture_every="
            f"{step_sync_capture_every} in {debug_summary_path}; using 1"
        )
        step_sync_capture_every = 1
    expected_video_frame_count = (
        (expected_step_sync_count + step_sync_capture_every - 1)
        // step_sync_capture_every
        if expected_step_sync_count is not None
        else None
    )
    step_sync_capture_count = _optional_int(debug_summary.get("step_sync_capture_count"))
    capture_count_source = "summary"
    if step_sync_capture_count is None:
        step_sync_capture_count = video_frame_count
        capture_count_source = "video_frame_count_legacy_fallback"
    match_count = _optional_int(debug_summary.get("step_sync_image_match_count"))
    reuse_count = _optional_int(debug_summary.get("step_sync_image_reuse_count"))
    placeholder_count = _optional_int(debug_summary.get("step_sync_placeholder_count"))
    policy_termination = _json_object(
        _first_present(result.get("policy_termination"), task_summary.get("policy_termination"))
    ) or {}
    no_fresh_action_count = _optional_int(result.get("no_fresh_action_count"))
    if no_fresh_action_count is None:
        no_fresh_action_count = _optional_int(
            policy_termination.get("total_no_fresh_action_count")
        )
    interaction_diagnostics = _interaction_diagnostics(attempts)
    mllm_summary = _mllm_summary(mllm_rows)
    mllm_summary["metrics_available"] = mllm_metrics_path is not None

    row = {
        "worker": artifacts.worker,
        "episode_index": _optional_int(
            _first_present(result.get("episode_index"), task_summary.get("episode_index"), plan.get("episode_index"))
        ),
        "house_index": _optional_int(
            _first_present(result.get("house_index"), task_summary.get("house_index"))
        ),
        "case_id": _first_present(result.get("case_id"), task_summary.get("case_id")),
        "status": (
            "incomplete"
            if artifacts.planned and completed is False
            else _first_present(
                result.get("status"), document.get("status"), task_summary.get("episode_status")
            )
        ),
        "completion": {
            "planned": artifacts.planned,
            "completed": completed,
            "result_present": result_path is not None,
            "evaluator_status": _first_present(result.get("status"), document.get("status")),
            "evaluator_success": _optional_bool(
                _first_present(result.get("success"), result.get("task_success"))
            ),
            "evaluator_terminal_reason": _first_present(
                result.get("terminal_reason"), task_summary.get("terminal_reason")
            ),
            "runner_exit_code": _optional_int(task_summary.get("runner_exit_code")),
            "timed_out": _optional_bool(task_summary.get("timed_out")),
            "error": task_summary.get("error"),
            "attempt": None if artifacts.attempt_dir is None else artifacts.attempt_dir.name,
        },
        "scoring_eligible": scoring_eligible,
        "domains": _string_list(_first_present(result.get("domains"), task_summary.get("domains"))),
        "interaction_requirement": interaction_requirement,
        "success": bool(
            completed is not False
            and _first_present(
                result.get("success"), result.get("task_success"), task_summary.get("success"), False
            )
        ),
        "outcomes": {
            "task_success": _optional_bool(
                _first_present(result.get("task_success"), task_summary.get("task_success"))
            ),
            "nav_success": _optional_bool(
                _first_present(result.get("nav_success"), task_summary.get("nav_success"))
            ),
            "required_interaction_success": _optional_bool(
                _first_present(
                    result.get("required_interaction_success"),
                    task_summary.get("required_interaction_success"),
                )
            ),
            "sequence_success": _optional_bool(
                _first_present(result.get("sequence_success"), task_summary.get("sequence_success"))
            ),
        },
        "terminal_reason": (
            "incomplete"
            if artifacts.planned and completed is False
            else _first_present(result.get("terminal_reason"), task_summary.get("terminal_reason"))
        ),
        "early_stop": {
            "triggered": bool(raw_early_stop.get("triggered")),
            "reason": raw_early_stop.get("reason"),
            "trigger_step": _optional_int(raw_early_stop.get("trigger_step")),
            "failed_subgoal_count": _optional_int(
                raw_early_stop.get("failed_subgoal_count")
            ),
            "observed_navigation_failure_count": _optional_int(
                raw_early_stop.get("observed_navigation_failure_count")
            ),
            "displacement_m": _finite_number(raw_early_stop.get("displacement_m")),
        },
        "step_count": expected_step_sync_count,
        "episode_step_budget": _optional_int(
            _first_present(result.get("episode_step_budget"), task_summary.get("max_steps"))
        ),
        "path": {
            "navigation_length_m": actual_path,
            "reference_length_m": reference_path,
            "reference_fraction": (
                float(actual_path) / float(reference_path)
                if actual_path is not None and reference_path not in (None, 0)
                else None
            ),
        },
        "target": {
            "distance_m": _finite_number(
                _first_present(result.get("target_distance_m"), task_summary.get("target_distance_m"))
            ),
            "visibility_fraction": _finite_number(
                _first_present(
                    result.get("target_visibility_fraction"),
                    task_summary.get("target_visibility_fraction"),
                )
            ),
        },
        "interactions": {
            "attempt_count": attempt_count,
            "correct_count": _optional_int(result.get("correct_interaction_action_count")) or 0,
            "repeat_count": _duplicate_interactions(attempts),
            "error_count": extra_count + invalid_count,
            "extra_count": extra_count,
            "invalid_count": invalid_count,
        },
        "interaction_diagnostics": interaction_diagnostics,
        "policy_termination": {
            "no_fresh_action_count": no_fresh_action_count,
            "applied_action_step_count": _optional_int(
                _first_present(
                    result.get("applied_action_step_count"),
                    task_summary.get("applied_action_step_count"),
                )
            ),
            "max_consecutive_no_fresh_action_count": _optional_int(
                policy_termination.get("max_consecutive_no_fresh_action_count")
            ),
            "max_no_fresh_wall_seconds": _finite_number(
                policy_termination.get("max_no_fresh_wall_seconds")
            ),
            "triggered": _optional_bool(policy_termination.get("triggered")),
            "reason": policy_termination.get("reason"),
            "source": policy_termination.get("source"),
        },
        # The formal paper metrics use evaluator-owned scalars.  SPL is the
        # sole exception: it is recomputed from saved NavToObj success and
        # planar paths so a stale interaction-conditioned `spl` field cannot
        # contaminate a post-hoc report.  A round summary must never infer
        # interaction correctness from redacted public attempts.
        "paper_metrics": {
            "schema_version": paper_metric_schema_version,
            "metric_config": paper_metric_config,
            "nav_success": paper_nav_success,
            "spl": paper_spl,
            "saved_spl_diagnostic": _finite_number(result.get("spl")),
            "required_interaction_success": _optional_bool(
                result.get("required_interaction_success")
            ),
            "interaction_precision": _finite_number(
                result.get("interaction_precision_episode")
            ),
            "total_cost": _finite_number(result.get("episode_total_cost")),
            "total_cost_breakdown": _json_object(
                result.get("episode_total_cost_breakdown")
            ),
            "interaction_attempt_count": _optional_int(
                result.get("interaction_action_count")
            ),
            "valid_interaction_attempt_count": _optional_int(
                result.get("valid_interaction_attempt_count")
            ),
            "error_interaction_attempt_count": _optional_int(
                result.get("error_interaction_attempt_count")
            ),
            "task_irrelevant_interaction_attempt_count": _optional_int(
                result.get("task_irrelevant_interaction_attempt_count")
            ),
            "failed_interaction_attempt_count": _optional_int(
                result.get("failed_interaction_attempt_count")
            ),
            "repeated_interaction_attempt_count": _optional_int(
                result.get("repeated_interaction_attempt_count")
            ),
        },
        "coverage": {
            "exploration_ratio": coverage_ratio,
            "source": coverage_source,
        },
        "rgb_step_sync": {
            "expected_step_sync_count": expected_step_sync_count,
            "step_sync_count": step_sync_count,
            "raw_step_sync_complete": (
                step_sync_count >= expected_step_sync_count
                if step_sync_count is not None and expected_step_sync_count is not None
                else None
            ),
            "step_sync_capture_every": step_sync_capture_every,
            "expected_video_frame_count": expected_video_frame_count,
            "step_sync_capture_count": step_sync_capture_count,
            "step_sync_capture_count_source": capture_count_source,
            "step_sync_capture_complete": (
                step_sync_capture_count >= expected_video_frame_count
                if step_sync_capture_count is not None and expected_video_frame_count is not None
                else None
            ),
            "video_frame_count": video_frame_count,
            "video_frame_count_complete": (
                video_frame_count >= expected_video_frame_count
                if video_frame_count is not None and expected_video_frame_count is not None
                else None
            ),
            "image_match_count": match_count,
            "image_reuse_count": reuse_count,
            "placeholder_count": placeholder_count,
            # Retain the old literal comparison for machine-readable backward
            # compatibility.  It is no longer a recorder-completeness check
            # when six-panel video is sampled below the raw step-sync rate.
            "video_frame_count_matches_step_sync": (
                step_sync_count == video_frame_count
                if step_sync_count is not None and video_frame_count is not None
                else None
            ),
        },
        "mllm": mllm_summary,
        "wall_time": {
            "evaluator_elapsed_s": _finite_number(
                _first_present(result.get("elapsed_seconds"), task_summary.get("elapsed_seconds"))
            ),
            "runner_elapsed_s": _finite_number(task_summary.get("elapsed_sec")),
            "recorder_duration_s": _finite_number(debug_summary.get("duration_sec")),
        },
        "artifacts": {
            "episode_result": None if result_path is None else str(result_path),
            "debug_summary": None if debug_summary_path is None else str(debug_summary_path),
            "mllm_metrics": None if mllm_metrics_path is None else str(mllm_metrics_path),
            "coverage": str(coverage_path) if coverage_path is not None else None,
        },
    }
    return row, warnings


def summarise_episode(round_root: Path, result_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Summarise one legacy evaluator result (kept for external callers)."""

    worker_dir = _worker_dir(round_root, result_path)
    attempt_dir = next(
        (
            parent
            for parent in result_path.parents
            if _ATTEMPT_DIR_RE.match(parent.name)
        ),
        None,
    )
    return _summarise_artifact(
        round_root,
        _EpisodeArtifacts(
            worker=worker_dir.name,
            task_dir=worker_dir,
            attempt_dir=attempt_dir,
            result_path=result_path,
            task_summary={},
            manifest_plan=None,
            planned=False,
        ),
    )


def _paper_metric_value(row: dict[str, Any], key: str) -> Any:
    paper = row.get("paper_metrics")
    return paper.get(key) if isinstance(paper, dict) else None


def _strict_paper_rate(
    rows: Sequence[dict[str, Any]], key: str
) -> tuple[float | None, int, int]:
    """Aggregate a Boolean metric only when every scored row persisted it."""

    denominator = len(rows)
    values = [_optional_bool(_paper_metric_value(row, key)) for row in rows]
    missing_count = sum(value is None for value in values)
    if denominator == 0 or missing_count:
        return None, denominator, missing_count
    return sum(bool(value) for value in values) / denominator, denominator, 0


def _strict_paper_mean(
    rows: Sequence[dict[str, Any]], key: str
) -> tuple[float | None, int, int]:
    """Aggregate a scalar metric without silently dropping incomplete rows."""

    denominator = len(rows)
    values = [_finite_number(_paper_metric_value(row, key)) for row in rows]
    missing_count = sum(value is None for value in values)
    if denominator == 0 or missing_count:
        return None, denominator, missing_count
    mean = sum(float(value) for value in values if value is not None) / denominator
    return mean, denominator, 0


def _paper_provenance(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, Any]:
    """Check that a group has one explicit evaluator metric contract."""

    serialized: dict[str, Any] = {}
    missing_count = 0
    for row in rows:
        value = _paper_metric_value(row, key)
        if value is None:
            missing_count += 1
            continue
        if key == "metric_config":
            if not isinstance(value, dict):
                missing_count += 1
                continue
            fingerprint = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            serialized[fingerprint] = value
        elif isinstance(value, str) and value:
            serialized[value] = value
        else:
            missing_count += 1

    consistent = bool(rows) and missing_count == 0 and len(serialized) == 1
    return {
        "value": next(iter(serialized.values())) if consistent else None,
        "consistent": consistent,
        "missing_count": missing_count,
        "distinct_count": len(serialized),
    }


def _paper_group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute the paper's five metrics from persisted evaluator scalars.

    This module deliberately does not reconstruct these values from action
    traces.  In particular, public interaction attempts do not contain the
    evaluator-private required/irrelevant/repeat classification needed for IP
    and Total Cost.
    """

    metric_schema = _paper_provenance(rows, "schema_version")
    metric_config = _paper_provenance(rows, "metric_config")
    sr, sr_denominator, sr_missing = _strict_paper_rate(rows, "nav_success")
    spl, spl_denominator, spl_missing = _strict_paper_mean(rows, "spl")
    required_rows = [
        row for row in rows if row.get("interaction_requirement") == "required"
    ]
    isr, isr_denominator, isr_missing = _strict_paper_rate(
        required_rows, "required_interaction_success"
    )
    ip, ip_denominator, ip_missing = _strict_paper_mean(
        rows, "interaction_precision"
    )
    total_cost, total_cost_denominator, total_cost_missing = _strict_paper_mean(
        rows, "total_cost"
    )

    # A legacy result can contain similarly named fields whose definitions
    # predate the paper protocol.  Do not present those as formal paper scores
    # unless every episode records one explicit, shared metric schema.
    if not metric_schema["consistent"]:
        sr = spl = isr = ip = total_cost = None

    # A Cost value without one frozen parameter vector is not comparable across
    # workers, so make it unavailable rather than averaging incompatible runs.
    if not metric_config["consistent"]:
        total_cost = None

    return {
        "episode_count": len(rows),
        "paper_metric_schema_version": metric_schema["value"],
        "paper_metric_schema_consistent": metric_schema["consistent"],
        "paper_metric_schema_missing_count": metric_schema["missing_count"],
        "paper_metric_schema_distinct_count": metric_schema["distinct_count"],
        "paper_metric_config": metric_config["value"],
        "paper_metric_config_consistent": metric_config["consistent"],
        "paper_metric_config_missing_count": metric_config["missing_count"],
        "paper_metric_config_distinct_count": metric_config["distinct_count"],
        # Canonical paper names.
        "sr": sr,
        "spl": spl,
        "isr": isr,
        "ip": ip,
        "total_cost": total_cost,
        # Compatibility aliases used by the evaluator's regular summary.
        "success_rate": sr,
        "mean_spl": spl,
        "required_interaction_success_rate": isr,
        "interaction_precision": ip,
        "mean_total_cost": total_cost,
        # Explicit denominators make N/A (for example ISR on unnecessary-only
        # episodes) distinguishable from a zero-valued metric.
        "sr_denominator": sr_denominator,
        "sr_missing_count": sr_missing,
        "spl_denominator": spl_denominator,
        "spl_missing_count": spl_missing,
        "isr_denominator": isr_denominator,
        "isr_missing_count": isr_missing,
        "ip_denominator": ip_denominator,
        "ip_missing_count": ip_missing,
        "total_cost_denominator": total_cost_denominator,
        "total_cost_missing_count": total_cost_missing,
    }


def _paper_round_summary(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible_rows = [
        row for row in episodes if bool(row.get("scoring_eligible", True))
    ]
    groups: dict[str, list[dict[str, Any]]] = {"overall": eligible_rows}
    for domain in ("channel", "container", "mixed"):
        expected = {"channel", "container"} if domain == "mixed" else {domain}
        groups[f"domain/{domain}"] = [
            row for row in eligible_rows if set(row.get("domains", [])) == expected
        ]
    for requirement in ("required", "unnecessary", "beneficial"):
        groups[f"requirement/{requirement}"] = [
            row
            for row in eligible_rows
            if row.get("interaction_requirement") == requirement
        ]

    overall = _paper_group_summary(eligible_rows)
    return {
        "schema_version": PAPER_METRIC_SCHEMA_VERSION,
        "scope": "scoring-eligible episodes discovered below round_root",
        "total_episode_count": len(episodes),
        "scoring_eligible_episode_count": len(eligible_rows),
        "runtime_ineligible_episode_count": len(episodes) - len(eligible_rows),
        "paper_metric_schema_version": overall["paper_metric_schema_version"],
        "paper_metric_schema_consistent": overall["paper_metric_schema_consistent"],
        "paper_metric_config": overall["paper_metric_config"],
        "paper_metric_config_consistent": overall["paper_metric_config_consistent"],
        "groups": {
            name: _paper_group_summary(rows)
            for name, rows in groups.items()
            if rows
        },
    }


def _paper_metric_warnings(paper_summary: dict[str, Any]) -> list[str]:
    groups = paper_summary.get("groups")
    overall = groups.get("overall") if isinstance(groups, dict) else None
    if not isinstance(overall, dict) or not overall.get("episode_count"):
        return []
    warnings: list[str] = []
    if not overall.get("paper_metric_schema_consistent"):
        warnings.append(
            "Paper metrics are not formally comparable: episode results do not "
            "share one persisted paper_metric_schema_version."
        )
    if not overall.get("paper_metric_config_consistent"):
        warnings.append(
            "Total Cost is unavailable: episode results do not share one persisted "
            "paper_metric_config."
        )
    for name, missing_key in (
        ("SR", "sr_missing_count"),
        ("SPL", "spl_missing_count"),
        ("ISR", "isr_missing_count"),
        ("IP", "ip_missing_count"),
        ("Total Cost", "total_cost_missing_count"),
    ):
        missing_count = int(overall.get(missing_key, 0) or 0)
        if missing_count:
            warnings.append(
                f"Paper {name} is unavailable for {missing_count} scoring-eligible episode(s) "
                "because the evaluator scalar was not persisted."
            )
    return warnings


def _completed_result_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return finished rows with a persisted evaluator result for diagnostics."""

    completed: list[dict[str, Any]] = []
    for row in rows:
        completion = row.get("completion")
        completion = completion if isinstance(completion, dict) else {}
        artifacts = row.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        if completion.get("completed") is False or not artifacts.get("episode_result"):
            continue
        completed.append(row)
    return completed


def _boolean_metric(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        _optional_bool((row.get("outcomes") or {}).get(key))
        for row in rows
        if isinstance(row.get("outcomes"), dict)
    ]
    true_count = sum(value is True for value in values)
    false_count = sum(value is False for value in values)
    missing_count = sum(value is None for value in values)
    denominator = true_count + false_count
    return {
        "success_count": true_count,
        "failure_count": false_count,
        "missing_count": missing_count,
        "denominator": denominator,
        "success_rate": true_count / denominator if denominator else None,
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _round_diagnostics(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = _completed_result_rows(episodes)
    planned_rows = [
        row
        for row in episodes
        if isinstance(row.get("completion"), dict) and bool(row["completion"].get("planned"))
    ]
    planned_count = len(planned_rows) if planned_rows else len(episodes)
    completed_count = sum(
        bool((row.get("completion") or {}).get("completed"))
        for row in planned_rows
    ) if planned_rows else len(completed_rows)
    incomplete_count = max(0, planned_count - completed_count)
    required_rows = [
        row for row in completed_rows if row.get("interaction_requirement") == "required"
    ]

    invalid_reason_counts: Counter[str] = Counter()
    failed_reason_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()
    policy_reason_counts: Counter[str] = Counter()
    policy_source_counts: Counter[str] = Counter()
    no_fresh_values: list[int] = []
    applied_values: list[int] = []
    max_consecutive_values: list[int] = []
    max_no_fresh_wall_values: list[float] = []
    m1_error_reason_counts: Counter[str] = Counter()
    for row in completed_rows:
        interaction = row.get("interaction_diagnostics")
        if isinstance(interaction, dict):
            invalid_reason_counts.update(
                {
                    str(reason): int(count)
                    for reason, count in (interaction.get("invalid_reason_counts") or {}).items()
                    if _optional_int(count) is not None
                }
            )
            failed_reason_counts.update(
                {
                    str(reason): int(count)
                    for reason, count in (interaction.get("failed_reason_counts") or {}).items()
                    if _optional_int(count) is not None
                }
            )
            unknown_reason_counts.update(
                {
                    str(reason): int(count)
                    for reason, count in (interaction.get("unknown_reason_counts") or {}).items()
                    if _optional_int(count) is not None
                }
            )
    for row in episodes:
        policy = row.get("policy_termination")
        if isinstance(policy, dict):
            if (value := _optional_int(policy.get("no_fresh_action_count"))) is not None:
                no_fresh_values.append(value)
            if (value := _optional_int(policy.get("applied_action_step_count"))) is not None:
                applied_values.append(value)
            if (value := _optional_int(policy.get("max_consecutive_no_fresh_action_count"))) is not None:
                max_consecutive_values.append(value)
            if (value := _finite_number(policy.get("max_no_fresh_wall_seconds"))) is not None:
                max_no_fresh_wall_values.append(float(value))
            if policy.get("reason"):
                policy_reason_counts[str(policy["reason"])] += 1
            if policy.get("source"):
                policy_source_counts[str(policy["source"])] += 1
        mllm = row.get("mllm")
        if isinstance(mllm, dict):
            m1_error_reason_counts.update(
                {
                    str(reason): int(count)
                    for reason, count in (mllm.get("m1_error_reason_counts") or {}).items()
                    if _optional_int(count) is not None
                }
            )

    return {
        "completion": {
            "planned_episode_count": planned_count,
            "completed_episode_count": completed_count,
            "incomplete_episode_count": incomplete_count,
            "completed_result_episode_count": len(completed_rows),
            "missing_result_episode_count": sum(
                not bool((row.get("artifacts") or {}).get("episode_result"))
                for row in planned_rows
            ),
        },
        "outcomes": {
            "task_success": _boolean_metric(completed_rows, "task_success"),
            "nav_success": _boolean_metric(completed_rows, "nav_success"),
            "required_interaction_success": _boolean_metric(
                required_rows, "required_interaction_success"
            ),
            "sequence_success": _boolean_metric(completed_rows, "sequence_success"),
        },
        "interactions": {
            "invalid_interaction_action_count": sum(
                int((row.get("interactions") or {}).get("invalid_count", 0) or 0)
                for row in completed_rows
            ),
            "unknown_interaction_attempt_count": sum(
                int((row.get("interaction_diagnostics") or {}).get("unknown_attempt_count", 0) or 0)
                for row in completed_rows
            ),
            "invalid_reason_counts": _counter_dict(invalid_reason_counts),
            "failed_interaction_attempt_count": sum(
                int((row.get("interaction_diagnostics") or {}).get("failed_interaction_attempt_count", 0) or 0)
                for row in completed_rows
            ),
            "failed_reason_counts": _counter_dict(failed_reason_counts),
            "unknown_reason_counts": _counter_dict(unknown_reason_counts),
        },
        "ros_policy": {
            "no_fresh_action_count": sum(no_fresh_values),
            "no_fresh_reported_episode_count": len(no_fresh_values),
            "episodes_with_no_fresh_action": sum(value > 0 for value in no_fresh_values),
            "max_consecutive_no_fresh_action_count": max(max_consecutive_values, default=None),
            "max_no_fresh_wall_seconds": max(max_no_fresh_wall_values, default=None),
            "applied_action_step_count": sum(applied_values),
            "applied_action_step_reported_episode_count": len(applied_values),
            "termination_reason_counts": _counter_dict(policy_reason_counts),
            "termination_source_counts": _counter_dict(policy_source_counts),
        },
        "m1": {
            "metrics_available_episode_count": sum(
                bool((row.get("mllm") or {}).get("metrics_available"))
                for row in episodes
            ),
            "call_count": sum(
                int((row.get("mllm") or {}).get("m1_call_count", 0) or 0)
                for row in episodes
            ),
            "error_count": sum(
                int((row.get("mllm") or {}).get("m1_error_count", 0) or 0)
                for row in episodes
            ),
            "error_reason_counts": _counter_dict(m1_error_reason_counts),
        },
    }


def summarise_round(round_root: Path) -> dict[str, Any]:
    """Build one machine-readable summary for a batch or legacy V3 round."""

    root = round_root.expanduser().resolve()
    episodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    artifacts = _batch_artifacts(root, warnings)
    if artifacts is None:
        artifacts = _legacy_artifacts(root)
    for artifact in artifacts:
        episode, episode_warnings = _summarise_artifact(root, artifact)
        episodes.append(episode)
        warnings.extend(episode_warnings)

    paper_metrics = _paper_round_summary(episodes)
    warnings.extend(_paper_metric_warnings(paper_metrics))
    diagnostics = _round_diagnostics(episodes)

    completed_result_rows = _completed_result_rows(episodes)
    success_count = sum(bool(row["success"]) for row in completed_result_rows)
    evaluator_times = [
        float(value)
        for row in episodes
        if (value := row["wall_time"]["evaluator_elapsed_s"]) is not None
    ]
    runner_times = [
        float(value)
        for row in episodes
        if (value := row["wall_time"].get("runner_elapsed_s")) is not None
    ]
    terminal_counts = Counter(
        str(
            row.get("terminal_reason")
            or ("incomplete" if (row.get("completion") or {}).get("completed") is False else "unknown")
        )
        for row in episodes
    )
    early_stop_rows = [
        row["early_stop"] for row in episodes if row["early_stop"]["triggered"]
    ]
    early_stop_counts = Counter(
        str(row.get("reason") or "unknown") for row in early_stop_rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "round_root": str(root),
        "episode_count": len(episodes),
        "completed_result_episode_count": len(completed_result_rows),
        "worker_count": len({str(row["worker"]) for row in episodes}),
        "success_count": success_count,
        "success_rate": (
            success_count / len(completed_result_rows)
            if completed_result_rows
            else None
        ),
        "round_diagnostics": diagnostics,
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "early_stop_count": len(early_stop_rows),
        "early_stop_reason_counts": dict(sorted(early_stop_counts.items())),
        "total_mllm_call_count": sum(int(row["mllm"]["call_count"]) for row in episodes),
        "total_pre_score_guard_count": sum(
            int(row["mllm"]["pre_score_guard_count"]) for row in episodes
        ),
        "total_repeated_candidate_count": sum(
            int(row["mllm"]["repeated_selected_candidate_count"]) for row in episodes
        ),
        "parallel_wall_time_estimate_s": max(evaluator_times, default=None),
        "episode_wall_time_sum_s": sum(evaluator_times) if evaluator_times else None,
        "parallel_runner_wall_time_estimate_s": max(runner_times, default=None),
        "episode_runner_wall_time_sum_s": sum(runner_times) if runner_times else None,
        "paper_metrics": paper_metrics,
        "episodes": episodes,
        "warnings": warnings,
    }


def _format_number(value: Any, digits: int = 2) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{float(number):.{digits}f}"


def _format_ratio(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{float(number):.1%}"


def _format_bool(value: Any) -> str:
    parsed = _optional_bool(value)
    return "Y" if parsed is True else "N" if parsed is False else "—"


def _table(rows: Iterable[dict[str, Any]]) -> str:
    headers = [
        "worker",
        "ep",
        "ok",
        "task/nav/req/seq",
        "terminal",
        "early-stop",
        "steps/budget",
        "path/ref m",
        "target/vis",
        "int C/R/E",
        "invalid/unknown",
        "no-fresh/applied",
        "coverage",
        "RGB match/place; raw/V",
        "MLLM/guard",
        "M1 err/calls",
        "cand repeat",
        "wall s",
    ]
    body: list[list[str]] = []
    for row in rows:
        rgb = row["rgb_step_sync"]
        match = rgb["image_match_count"]
        match_text = "—" if match is None else str(match)
        placeholder = rgb["placeholder_count"]
        raw_count = rgb["step_sync_count"]
        raw_expected = rgb["expected_step_sync_count"]
        video_count = rgb["video_frame_count"]
        video_expected = rgb["expected_video_frame_count"]
        raw_text = (
            f"{raw_count}/{raw_expected}"
            if raw_count is not None and raw_expected is not None
            else "—"
        )
        video_text = (
            f"{video_count}/{video_expected}"
            if video_count is not None and video_expected is not None
            else "—"
        )
        early_stop = row["early_stop"]
        early_stop_text = "—"
        if early_stop["triggered"]:
            reason = str(early_stop["reason"] or "unknown")
            failed_count = early_stop["failed_subgoal_count"]
            trigger_step = early_stop["trigger_step"]
            early_stop_text = (
                f"{reason}; n={failed_count if failed_count is not None else '—'}"
                f"@{trigger_step if trigger_step is not None else '—'}"
            )
        outcomes = row.get("outcomes") or {}
        policy_termination = row.get("policy_termination") or {}
        interaction_diagnostics = row.get("interaction_diagnostics") or {}
        body.append(
            [
                str(row["worker"]),
                str(row["episode_index"] if row["episode_index"] is not None else "—"),
                "Y" if row["success"] else "N",
                "/".join(
                    _format_bool(outcomes.get(key))
                    for key in (
                        "task_success",
                        "nav_success",
                        "required_interaction_success",
                        "sequence_success",
                    )
                ),
                str(
                    row["terminal_reason"]
                    or (
                        "incomplete"
                        if (row.get("completion") or {}).get("completed") is False
                        else "—"
                    )
                ),
                early_stop_text,
                f"{row['step_count'] if row['step_count'] is not None else '—'}/"
                f"{row['episode_step_budget'] if row['episode_step_budget'] is not None else '—'}",
                f"{_format_number(row['path']['navigation_length_m'])}/"
                f"{_format_number(row['path']['reference_length_m'])}",
                f"{_format_number(row['target']['distance_m'])}/"
                f"{_format_ratio(row['target']['visibility_fraction'])}",
                f"{row['interactions']['correct_count']}/"
                f"{row['interactions']['repeat_count']}/"
                f"{row['interactions']['error_count']}",
                f"{row['interactions']['invalid_count']}/"
                f"{interaction_diagnostics.get('unknown_attempt_count', 0)}",
                f"{policy_termination.get('no_fresh_action_count') if policy_termination.get('no_fresh_action_count') is not None else '—'}/"
                f"{policy_termination.get('applied_action_step_count') if policy_termination.get('applied_action_step_count') is not None else '—'}",
                _format_ratio(row["coverage"]["exploration_ratio"]),
                f"{match_text}/{placeholder if placeholder is not None else '—'}; "
                f"{raw_text}/{video_text}",
                f"{row['mllm']['call_count']}/{row['mllm']['pre_score_guard_count']}",
                f"{row['mllm']['m1_error_count']}/{row['mllm']['m1_call_count']}",
                str(row["mllm"]["repeated_selected_candidate_count"]),
                _format_number(row["wall_time"]["evaluator_elapsed_s"], digits=1),
            ]
        )

    widths = [len(header) for header in headers]
    for cells in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]

    def render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(cells) for cells in body)])


def render_terminal_summary(summary: dict[str, Any]) -> str:
    table = _table(summary.get("episodes", []))
    episode_count = int(summary.get("episode_count", 0) or 0)
    completed_result_count = int(
        summary.get("completed_result_episode_count", episode_count) or 0
    )
    success_count = int(summary.get("success_count", 0) or 0)
    rate = _format_ratio(summary.get("success_rate"))
    parallel_wall = _format_number(summary.get("parallel_wall_time_estimate_s"), digits=1)
    parallel_runner_wall = _format_number(
        summary.get("parallel_runner_wall_time_estimate_s"), digits=1
    )
    diagnostics = summary.get("round_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    completion = diagnostics.get("completion") if isinstance(diagnostics.get("completion"), dict) else {}
    outcomes = diagnostics.get("outcomes") if isinstance(diagnostics.get("outcomes"), dict) else {}
    interactions = diagnostics.get("interactions") if isinstance(diagnostics.get("interactions"), dict) else {}
    ros_policy = diagnostics.get("ros_policy") if isinstance(diagnostics.get("ros_policy"), dict) else {}
    m1 = diagnostics.get("m1") if isinstance(diagnostics.get("m1"), dict) else {}
    totals = (
        f"episodes={episode_count} planned={completion.get('planned_episode_count', episode_count)} "
        f"completed={completion.get('completed_episode_count', episode_count)} "
        f"incomplete={completion.get('incomplete_episode_count', 0)} "
        f"workers={summary.get('worker_count', 0)} "
        f"success={success_count}/{completed_result_count} ({rate}) "
        f"task={_format_ratio((outcomes.get('task_success') or {}).get('success_rate'))} "
        f"nav={_format_ratio((outcomes.get('nav_success') or {}).get('success_rate'))} "
        f"required={_format_ratio((outcomes.get('required_interaction_success') or {}).get('success_rate'))} "
        f"sequence={_format_ratio((outcomes.get('sequence_success') or {}).get('success_rate'))} "
        f"mllm={summary.get('total_mllm_call_count', 0)} "
        f"m1_errors={m1.get('error_count', 0)}/{m1.get('call_count', 0)} "
        f"invalid={interactions.get('invalid_interaction_action_count', 0)} "
        f"unknown={interactions.get('unknown_interaction_attempt_count', 0)} "
        f"no_fresh={ros_policy.get('no_fresh_action_count', 0)} "
        f"applied={ros_policy.get('applied_action_step_count', 0)} "
        f"pre_score_guard={summary.get('total_pre_score_guard_count', 0)} "
        f"candidate_repeats={summary.get('total_repeated_candidate_count', 0)} "
        f"early_stops={summary.get('early_stop_count', 0)} "
        f"early_stop_reasons={summary.get('early_stop_reason_counts', {})} "
        f"parallel_wall_estimate={parallel_wall}s "
        f"parallel_runner_wall_estimate={parallel_runner_wall}s"
    )
    warning_count = len(summary.get("warnings", []))
    if warning_count:
        totals += f" warnings={warning_count}"
    paper_metrics = summary.get("paper_metrics")
    paper_groups = (
        paper_metrics.get("groups") if isinstance(paper_metrics, dict) else None
    )
    paper_overall = (
        paper_groups.get("overall") if isinstance(paper_groups, dict) else None
    )
    if isinstance(paper_overall, dict):
        totals += (
            " paper="
            f"SR={_format_ratio(paper_overall.get('sr'))} "
            f"SPL={_format_ratio(paper_overall.get('spl'))} "
            f"ISR={_format_ratio(paper_overall.get('isr'))} "
            f"IP={_format_ratio(paper_overall.get('ip'))} "
            f"TotalCost={_format_number(paper_overall.get('total_cost'))}"
        )
    return f"{table}\n{totals}" if table else totals


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_root", type=Path, help="Root containing worker evaluation directories")
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="stdout representation (default: table)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally write the complete JSON summary to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = summarise_round(args.round_root)
    except NotADirectoryError as exc:
        raise SystemExit(str(exc)) from exc

    serialized = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized, encoding="utf-8")
    if args.format == "json":
        print(serialized, end="")
    else:
        print(render_terminal_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
