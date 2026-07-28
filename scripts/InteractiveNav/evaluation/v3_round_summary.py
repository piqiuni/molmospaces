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
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "interactive_nav_v3_round_summary_v1"
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
    result_path: Path,
    worker_dir: Path,
    debug_summary: dict[str, Any],
    warnings: list[str],
) -> tuple[float | int | None, str | None, Path | None]:
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
    return {
        "call_count": len(rows),
        "successful_call_count": len(rows) - error_count,
        "error_count": error_count,
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


def summarise_episode(round_root: Path, result_path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    document = _read_json(result_path, warnings)
    result = _result_payload(document)
    worker_dir = _worker_dir(round_root, result_path)
    debug_summary_path = worker_dir / "debug" / "summary.json"
    debug_summary = (
        _read_json(debug_summary_path, warnings) if debug_summary_path.is_file() else {}
    )
    mllm_metrics_path = worker_dir / "mllm_metrics.jsonl"
    mllm_rows = _read_jsonl(mllm_metrics_path, warnings) if mllm_metrics_path.is_file() else []
    coverage_ratio, coverage_source, coverage_path = _coverage_payload(
        result_path, worker_dir, debug_summary, warnings
    )

    actual_path = _finite_number(result.get("navigation_path_length_m"))
    reference_path = _finite_number(result.get("reference_path_length_m"))
    attempts = result.get("interaction_attempts")
    attempt_count = len(attempts) if isinstance(attempts, list) else 0
    extra_count = _optional_int(result.get("extra_interaction_action_count")) or 0
    invalid_count = _optional_int(result.get("invalid_interaction_action_count")) or 0

    expected_step_sync_count = _optional_int(result.get("step_count"))
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

    row = {
        "worker": worker_dir.name,
        "episode_index": _optional_int(result.get("episode_index")),
        "house_index": _optional_int(result.get("house_index")),
        "case_id": result.get("case_id"),
        "status": result.get("status", document.get("status")),
        "scoring_eligible": result.get("scoring_eligible"),
        "success": bool(result.get("success", result.get("task_success", False))),
        "terminal_reason": result.get("terminal_reason"),
        "step_count": expected_step_sync_count,
        "episode_step_budget": _optional_int(result.get("episode_step_budget")),
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
            "distance_m": _finite_number(result.get("target_distance_m")),
            "visibility_fraction": _finite_number(result.get("target_visibility_fraction")),
        },
        "interactions": {
            "attempt_count": attempt_count,
            "correct_count": _optional_int(result.get("correct_interaction_action_count")) or 0,
            "repeat_count": _duplicate_interactions(attempts),
            "error_count": extra_count + invalid_count,
            "extra_count": extra_count,
            "invalid_count": invalid_count,
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
        "mllm": _mllm_summary(mllm_rows),
        "wall_time": {
            "evaluator_elapsed_s": _finite_number(result.get("elapsed_seconds")),
            "recorder_duration_s": _finite_number(debug_summary.get("duration_sec")),
        },
        "artifacts": {
            "episode_result": str(result_path),
            "debug_summary": str(debug_summary_path) if debug_summary_path.is_file() else None,
            "mllm_metrics": str(mllm_metrics_path) if mllm_metrics_path.is_file() else None,
            "coverage": str(coverage_path) if coverage_path is not None else None,
        },
    }
    return row, warnings


def summarise_round(round_root: Path) -> dict[str, Any]:
    """Build the machine-readable summary for all discovered episodes."""

    root = round_root.expanduser().resolve()
    result_paths = discover_episode_results(root)
    episodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    for result_path in result_paths:
        episode, episode_warnings = summarise_episode(root, result_path)
        episodes.append(episode)
        warnings.extend(episode_warnings)

    success_count = sum(bool(row["success"]) for row in episodes)
    evaluator_times = [
        float(value)
        for row in episodes
        if (value := row["wall_time"]["evaluator_elapsed_s"]) is not None
    ]
    terminal_counts = Counter(str(row.get("terminal_reason") or "unknown") for row in episodes)
    return {
        "schema_version": SCHEMA_VERSION,
        "round_root": str(root),
        "episode_count": len(episodes),
        "worker_count": len({str(row["worker"]) for row in episodes}),
        "success_count": success_count,
        "success_rate": success_count / len(episodes) if episodes else None,
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "total_mllm_call_count": sum(int(row["mllm"]["call_count"]) for row in episodes),
        "total_pre_score_guard_count": sum(
            int(row["mllm"]["pre_score_guard_count"]) for row in episodes
        ),
        "total_repeated_candidate_count": sum(
            int(row["mllm"]["repeated_selected_candidate_count"]) for row in episodes
        ),
        "parallel_wall_time_estimate_s": max(evaluator_times, default=None),
        "episode_wall_time_sum_s": sum(evaluator_times) if evaluator_times else None,
        "episodes": episodes,
        "warnings": warnings,
    }


def _format_number(value: Any, digits: int = 2) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{float(number):.{digits}f}"


def _format_ratio(value: Any) -> str:
    number = _finite_number(value)
    return "—" if number is None else f"{float(number):.1%}"


def _table(rows: Iterable[dict[str, Any]]) -> str:
    headers = [
        "worker",
        "ep",
        "ok",
        "terminal",
        "steps/budget",
        "path/ref m",
        "target/vis",
        "int C/R/E",
        "coverage",
        "RGB match/place; raw/V",
        "MLLM/guard",
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
        body.append(
            [
                str(row["worker"]),
                str(row["episode_index"] if row["episode_index"] is not None else "—"),
                "Y" if row["success"] else "N",
                str(row["terminal_reason"] or "—"),
                f"{row['step_count'] if row['step_count'] is not None else '—'}/"
                f"{row['episode_step_budget'] if row['episode_step_budget'] is not None else '—'}",
                f"{_format_number(row['path']['navigation_length_m'])}/"
                f"{_format_number(row['path']['reference_length_m'])}",
                f"{_format_number(row['target']['distance_m'])}/"
                f"{_format_ratio(row['target']['visibility_fraction'])}",
                f"{row['interactions']['correct_count']}/"
                f"{row['interactions']['repeat_count']}/"
                f"{row['interactions']['error_count']}",
                _format_ratio(row["coverage"]["exploration_ratio"]),
                f"{match_text}/{placeholder if placeholder is not None else '—'}; "
                f"{raw_text}/{video_text}",
                f"{row['mllm']['call_count']}/{row['mllm']['pre_score_guard_count']}",
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
    success_count = int(summary.get("success_count", 0) or 0)
    rate = _format_ratio(summary.get("success_rate"))
    parallel_wall = _format_number(summary.get("parallel_wall_time_estimate_s"), digits=1)
    totals = (
        f"episodes={episode_count} workers={summary.get('worker_count', 0)} "
        f"success={success_count}/{episode_count} ({rate}) "
        f"mllm={summary.get('total_mllm_call_count', 0)} "
        f"pre_score_guard={summary.get('total_pre_score_guard_count', 0)} "
        f"candidate_repeats={summary.get('total_repeated_candidate_count', 0)} "
        f"parallel_wall_estimate={parallel_wall}s"
    )
    warning_count = len(summary.get("warnings", []))
    if warning_count:
        totals += f" warnings={warning_count}"
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
