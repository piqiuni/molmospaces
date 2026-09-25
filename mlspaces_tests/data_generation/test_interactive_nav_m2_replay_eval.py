from __future__ import annotations

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from scripts.InteractiveNav.evaluation import m2_replay_eval


def _write_fixture_attempt(root: Path) -> Path:
    attempt = root / "episode_0001" / "attempt_001"
    attempt.mkdir(parents=True)
    metrics = {
        "timestamp": 10.0,
        "role": "subgoal_selection",
        "model": "recorded-model",
        "episode_id": "episode_fixture",
        "graph_revision": 7,
        "candidate_sequence": 9,
        "candidate_ids": ["frontier:1"],
        "candidate_pool_count": 1,
        "curated_candidate_count": 1,
        "candidate_options": [
            {
                "id": "frontier:1",
                "action": "explore",
                "subject_id": "frontier_1",
                "subject_type": "frontier",
                "effect": "reveal_space",
                "distance_m": 1.0,
                "room_id": "room_1",
                "pre_score": 1.0,
            }
        ],
        "selection_granularity": "candidate",
        "raw_text": json.dumps(
            {
                "ranked_ids": ["frontier:1"],
                "reason": "INFORMATION_GAIN",
                "confidence": "medium",
            }
        ),
        "error": "",
    }
    (attempt / "mllm_metrics.jsonl").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8"
    )

    boundary = {
        "step_index": 4,
        # This private field is intentionally present in the source fixture.  It
        # must never be copied into the replay case.
        "gt_observations": [{"instance_id": "private_target"}],
        "semantic_candidates": {
            "schema_version": 1,
            "sequence": 9,
            "episode_id": "episode_fixture",
            "graph_revision": 7,
            "robot_xy": [0.0, 0.0],
            "target_context": {
                "schema_version": 1,
                "episode_id": "episode_fixture",
                "enabled": True,
                "target_name": "egg",
                "object_labels": ["egg"],
                "instruction": "find the egg",
                # Even if an older source accidentally retained a private key,
                # the production compact projection must discard it.
                "target_instance_id": "private_target",
            },
            "exploration_context": {
                "ready": True,
                "observation_step": 4,
            },
            "candidates": [
                {
                    "candidate_id": "frontier:1",
                    "behavior_type": "EXPLORE",
                    "source": "frontier",
                    "target_id": "frontier_1",
                    "target_name": "frontier",
                    "goal_xyyaw": [1.0, 0.0, 0.0],
                    "interaction_command": None,
                    "features": {"distance_m": 1.0},
                    "metadata": {
                        "cell_count": 10,
                        "map_resolution": 0.1,
                        "expected_visible_unknown_area_m2": 3.0,
                    },
                    "score": 1.0,
                    "score_terms": {},
                }
            ],
        },
        "unified_graph": {
            "episode_id": "episode_fixture",
            "graph_revision": 7,
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "label": "kitchen",
                    "centroid": [0.0, 0.0, 0.0],
                    "attributes": {
                        "active": True,
                        "room_attribute": "kitchen",
                        "room_attribute_confidence": 0.9,
                    },
                }
            ],
            "edges": [],
        },
    }
    raw = attempt / "debug" / "raw"
    raw.mkdir(parents=True)
    with gzip.open(raw / "step_boundaries.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(boundary) + "\n")
    return attempt


def test_extract_cases_reconstructs_public_request_without_private_gt(tmp_path: Path) -> None:
    _write_fixture_attempt(tmp_path)

    cases, warnings = m2_replay_eval.extract_cases(tmp_path, strict=True)

    assert warnings == []
    assert len(cases) == 1
    case = cases[0]
    assert case["request"]["mission"]["target"]["name"] == "egg"
    assert case["request"]["candidates"][0]["id"] == "frontier:1"
    assert case["reference"]["response"]["ranked_ids"] == ["frontier:1"]
    assert case["reference"]["is_correctness_label"] is False
    assert case["reconstruction"] == {
        "mode": "reconstructed_public_context",
        "missing_fields": [
            "recent_decisions",
            "entered_room_ids",
            "candidate_history",
        ],
    }
    serialized_request = json.dumps(case["request"])
    assert "private_target" not in serialized_request
    assert "gt_observations" not in serialized_request


def test_audit_public_request_rejects_private_target_fields() -> None:
    request = {
        "schema_version": 4,
        "instruction": "rank",
        "mission": {"target_instance_id": "private"},
        "robot": {},
        "recent_decisions": [],
        "graph": {},
        "room_object_reasoning": {},
        "candidates": [{"id": "frontier:1"}],
    }

    with pytest.raises(m2_replay_eval.DatasetError, match="evaluator-private"):
        m2_replay_eval.audit_public_request(request)


def test_replay_retries_transport_error_and_scores_annotation() -> None:
    case = {
        "schema_version": m2_replay_eval.CASE_SCHEMA_VERSION,
        "case_id": "case-1",
        "request": {
            "schema_version": 4,
            "instruction": "rank",
            "mission": {},
            "robot": {},
            "recent_decisions": [],
            "graph": {},
            "room_object_reasoning": {},
            "candidates": [{"id": "frontier:1"}, {"id": "frontier:2"}],
        },
        "reference": {
            "response": {
                "ranked_ids": ["frontier:2", "frontier:1"],
                "reason": "INFORMATION_GAIN",
                "confidence": "medium",
            }
        },
        "annotation": {
            "status": "reviewed",
            "acceptable_top1_ids": ["frontier:1"],
            "preferred_ranking": ["frontier:1"],
            "forbidden_ids": ["frontier:2"],
        },
    }
    calls = 0

    def request_function(request: object, case_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None, {"error": "timed out", "latency_s": 1.0}
        return (
            {
                "ranked_ids": ["frontier:1", "frontier:2"],
                "reason": "INFORMATION_GAIN",
                "confidence": "high",
            },
            {"error": "", "latency_s": 0.5, "prompt_tokens": 20},
        )

    predictions = m2_replay_eval.replay_cases(
        [case], request_function=request_function, max_retries=1
    )
    summary = m2_replay_eval.summarize_scores(predictions)

    assert calls == 2
    assert predictions[0]["retry_count"] == 1
    assert [item["attempt"] for item in predictions[0]["attempts"]] == [1, 2]
    assert predictions[0]["score"]["acceptable_top1"] is True
    assert predictions[0]["score"]["forbidden_top1"] is False
    assert summary["valid_response_rate"] == 1.0
    assert summary["acceptable_top1_accuracy"] == 1.0
    assert summary["recorded_top1_agreement"] == 0.0
    assert predictions[0]["latency_s"] == 1.5
    assert predictions[0]["attempts"][0]["latency_s"] == 1.0


def test_replay_does_not_retry_non_timeout_error() -> None:
    case = {
        "schema_version": m2_replay_eval.CASE_SCHEMA_VERSION,
        "case_id": "case-no-retry",
        "request": {
            "schema_version": 4,
            "instruction": "rank",
            "mission": {},
            "robot": {},
            "recent_decisions": [],
            "graph": {},
            "room_object_reasoning": {},
            "candidates": [{"id": "frontier:1"}],
        },
        "reference": {},
        "annotation": {},
    }
    calls = 0

    def request_function(request: object, case_id: str):
        nonlocal calls
        calls += 1
        return None, {"error": "HTTP 401: unauthorized", "latency_s": 0.1}

    prediction = m2_replay_eval.replay_cases(
        [case], request_function=request_function, max_retries=3
    )[0]

    assert calls == 1
    assert prediction["retry_count"] == 0
    assert len(prediction["attempts"]) == 1


def test_validate_response_rejects_historical_or_hallucinated_id() -> None:
    response, error = m2_replay_eval.validate_response(
        {
            "ranked_ids": ["frontier:old"],
            "reason": "INFORMATION_GAIN",
            "confidence": "low",
        },
        ["frontier:current"],
    )

    assert response is None
    assert error == "ranked_ids_not_in_current_candidates:frontier:old"


def _replay_case(case_id: str) -> dict:
    return {
        "schema_version": m2_replay_eval.CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "request": {
            "schema_version": 4,
            "instruction": "rank",
            "mission": {},
            "robot": {},
            "recent_decisions": [],
            "graph": {},
            "room_object_reasoning": {},
            "candidates": [{"id": "frontier:1"}],
        },
        "annotation": {"status": "reviewed", "acceptable_top1_ids": ["frontier:1"]},
    }


def _valid_response() -> dict:
    return {
        "ranked_ids": ["frontier:1"],
        "reason": "INFORMATION_GAIN",
        "confidence": "high",
    }


def test_parallel_replay_reports_completion_before_slow_case_and_returns_input_order() -> None:
    both_running = threading.Barrier(2)
    slow_release = threading.Event()
    completed: list[str] = []
    coordinator_id = threading.get_ident()

    def request_function(request: object, case_id: str):
        both_running.wait(timeout=5)
        if case_id == "slow":
            assert slow_release.wait(timeout=5)
        return _valid_response(), {"latency_s": 0.1}

    def on_prediction(prediction: dict) -> None:
        assert threading.get_ident() == coordinator_id
        completed.append(prediction["case_id"])
        if prediction["case_id"] == "fast":
            slow_release.set()

    predictions = m2_replay_eval.replay_cases(
        [_replay_case("slow"), _replay_case("fast")],
        request_function=request_function,
        concurrency=2,
        on_prediction=on_prediction,
    )

    assert completed == ["fast", "slow"]
    assert [row["case_id"] for row in predictions] == ["slow", "fast"]
    assert all(row["score"]["acceptable_top1"] is True for row in predictions)


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.lock = threading.Lock()

    def monotonic(self) -> float:
        with self.lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.now += seconds


def _assert_rolling_request_limit(timestamps: list[float], limit: int) -> None:
    for window_start in timestamps:
        assert sum(window_start <= stamp < window_start + 60.0 for stamp in timestamps) <= limit


def test_rate_limiter_is_thread_safe_and_bounds_every_rolling_window() -> None:
    clock = _VirtualClock()
    limiter = m2_replay_eval.SlidingWindowRateLimiter(
        2, clock=clock.monotonic, sleep=clock.sleep,
    )
    barrier = threading.Barrier(8)

    def acquire() -> float:
        barrier.wait(timeout=5)
        return limiter.acquire()

    with ThreadPoolExecutor(max_workers=8) as pool:
        waits = list(pool.map(lambda _: acquire(), range(8)))

    timestamps = limiter.admission_timestamps
    assert len(timestamps) == 8
    assert any(wait > 0 for wait in waits)
    _assert_rolling_request_limit(timestamps, 2)


def test_shared_limiter_counts_retries_across_model_replays() -> None:
    clock = _VirtualClock()
    limiter = m2_replay_eval.SlidingWindowRateLimiter(
        2, clock=clock.monotonic, sleep=clock.sleep,
    )
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def request_function(request: object, case_id: str):
        with lock:
            calls[case_id] = calls.get(case_id, 0) + 1
            attempt = calls[case_id]
        if attempt == 1:
            return None, {"error": "timed out", "latency_s": 1.0}
        return _valid_response(), {"latency_s": 0.5}

    predictions = m2_replay_eval.replay_cases(
        [_replay_case("model-a-1"), _replay_case("model-a-2")],
        request_function=request_function, concurrency=2, max_retries=1,
        rate_limiter=limiter,
    )
    predictions.extend(m2_replay_eval.replay_cases(
        [_replay_case("model-b-1")], request_function=request_function,
        max_retries=1, rate_limiter=limiter,
    ))

    assert len(limiter.admission_timestamps) == 6
    _assert_rolling_request_limit(limiter.admission_timestamps, 2)
    assert all(row["retry_count"] == 1 for row in predictions)
    assert all(row["latency_s"] == 1.5 for row in predictions)
    assert any(row["rate_limit_wait_s"] >= 60.0 for row in predictions)
    assert m2_replay_eval.summarize_scores(predictions)["request_attempt_count"] == 6


def test_failed_labeled_cases_remain_in_accuracy_denominator() -> None:
    cases = [_replay_case(case_id) for case_id in ("good", "timeout", "invalid", "unlabeled")]
    cases[-1]["annotation"] = {}

    def request_function(request: object, case_id: str):
        if case_id == "timeout":
            raise TimeoutError("deadline exceeded")
        if case_id == "invalid":
            return {**_valid_response(), "ranked_ids": ["invented-id"]}, {}
        return _valid_response(), {}

    predictions = m2_replay_eval.replay_cases(
        cases, request_function=request_function, concurrency=4,
    )
    summary = m2_replay_eval.summarize_scores(predictions)

    assert summary["acceptable_top1_accuracy"] == pytest.approx(1 / 3)
    assert summary["acceptable_top1_case_count"] == 3
    assert summary["transport_error_count"] == 1
    assert [row["score"]["acceptable_top1"] for row in predictions] == [True, False, False, None]


def test_replay_cli_journals_predictions_as_they_complete(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "replay"
    cases = [_replay_case("one"), _replay_case("two")]
    m2_replay_eval.write_dataset(dataset_dir, cases)

    def build_request(args, directory):
        def request_function(request: object, case_id: str):
            if case_id == "two":
                rows = list(m2_replay_eval._jsonl_rows(output_dir / "predictions.inprogress.jsonl"))
                assert [row["case_id"] for row in rows] == ["one"]
            return _valid_response(), {}
        return request_function

    # Sequential completion callbacks are observable before the next request;
    # concurrent mode has its own ordering/callback regression above.
    monkeypatch.setattr(m2_replay_eval, "build_mllm_request_function", build_request)
    assert m2_replay_eval.main([
        "replay", "--dataset", str(dataset_dir), "--output-dir", str(output_dir),
        "--concurrency", "1", "--requests-per-minute", "0",
    ]) == 0
    final_rows = list(m2_replay_eval._jsonl_rows(output_dir / "predictions.jsonl"))
    journal_rows = list(m2_replay_eval._jsonl_rows(output_dir / "predictions.inprogress.jsonl"))
    assert final_rows == journal_rows
    assert len(journal_rows) == 2
    progress_lines = capsys.readouterr().err.splitlines()
    assert [json.loads(line)["completed"] for line in progress_lines] == [1, 2]
