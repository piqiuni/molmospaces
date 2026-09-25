import json
import time

from scripts.InteractiveNav.evaluation import collect_timeout_performance as collector
from scripts.InteractiveNav.evaluation import monitor_mllm_timeouts as monitor


def test_monitor_handles_single_vllm_partial_lines_and_log_rotation(tmp_path):
    path = tmp_path / "vllm.launcher.log"
    response = 'POST /v1/chat/completions HTTP/1.1" 200 OK'
    path.write_text(response)
    backend = monitor.BackendLogMonitor(tmp_path)
    assert "HTTP200累计=0" in backend.report()
    with path.open("a") as stream:
        stream.write("\n" + response + "\n")
    assert "HTTP200累计=2 增量=2" in backend.report()
    assert "增量=0" in backend.report()
    path.write_text(response + "\n")
    assert "HTTP200累计=1 增量=1" in backend.report()


def test_monitor_supports_single_and_replica_log_names(tmp_path):
    for name in ("vllm", "gpu2", "gpu10", "lb"):
        (tmp_path / f"{name}.launcher.log").write_text("\n")
    report = monitor.BackendLogMonitor(tmp_path).report()
    assert report.index("gpu2:") < report.index("gpu10:") < report.index("vllm:")
    assert "lb:" not in report


def test_timeout_roles_and_attempt_scoped_object_counts(tmp_path):
    attempts = []
    for index in (1, 2):
        attempt = tmp_path / "episode_0001" / f"attempt_{index}"
        attempt.mkdir(parents=True)
        attempts.append(attempt)
        rows = [
            {"role": "attribute_inference", "object_id": "fridge", "episode_id": "same",
             "timestamp": time.time(), "latency_s": 30, "is_timeout": True}
            for _ in range(6)
        ]
        rows.append({"role": "room_attribute_inference", "error": "deadline exceeded"})
        (attempt / "mllm_metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + '\n[]\n{"partial":'
        )
    report = collector._model_metrics(attempts)
    assert report["by_role"]["attribute_inference"]["timeouts"] == 12
    assert report["by_role"]["attribute_inference"]["errors"] == 12
    assert report["by_role"]["room_attribute_inference"]["timeouts"] == 2
    assert report["m1_max_calls_per_object"] == 6
    assert report["m1_objects_above_ten_calls"] == 0
    groups, recent, count, malformed, completed, succeeded = monitor.summarize(tmp_path, 120)
    assert groups["M1-objects"]["timeouts"] == 12
    assert recent["M1-objects"]["calls"] == 12
    assert groups["M1-rooms"]["timeouts"] == 2
    assert (count, malformed, completed, succeeded) == (2, 2, 0, 0)
