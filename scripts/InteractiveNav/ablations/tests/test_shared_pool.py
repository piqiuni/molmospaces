import json
from pathlib import Path
import sys
import threading
import time

import pytest

from ablations.report_large_pool import summarize
from ablations.preflight import check_model_endpoints
from run_ablation_pool import drain_pool, main, select_scenes


def test_shared_pool_refills_free_slot_across_variants_without_waiting_for_slow_slot():
    events = []
    release = threading.Event()

    def execute(worker, job):
        events.append(("start", worker, job))
        if job == "flat":
            assert release.wait(timeout=2.)
        elif job == "greedy":
            time.sleep(.02)
        else:
            release.set()
        events.append(("finish", worker, job))

    drain_pool(["flat", "greedy", "perception"], 2, execute, threading.Event())
    assignments = {job: worker for kind, worker, job in events if kind == "start"}
    assert assignments["greedy"] == assignments["perception"] != assignments["flat"]
    assert events.index(("start", assignments["perception"], "perception")) < events.index(("finish", assignments["flat"], "flat"))


def test_selection_excludes_invalid_and_prefers_success_before_partial_interaction():
    rows = [{"episode_index": index, "completed": True, "scoring_eligible": True,
             "success": index == 2001, "correct_interaction_action_count": 2,
             "interaction_action_count": 2} for index in (1000, 2000, 2001, 2002)]
    rows[-1]["scoring_eligible"] = False
    assert [r["episode_index"] for r in select_scenes({"episodes": rows}, 2)] == [2001, 2000]


def test_explicit_selection_uses_only_frozen_mixed_indices():
    rows = [{"episode_index": index, "completed": True, "scoring_eligible": True}
            for index in (2431, 2012, 2999)]
    assert [r["episode_index"] for r in select_scenes(
        {"episodes": rows}, 3, selection_mode="explicit")] == [2012, 2431, 2999]
    with pytest.raises(ValueError):
        select_scenes({"episodes": rows + [rows[0]]}, 4, selection_mode="explicit")
    with pytest.raises(ValueError):
        select_scenes({"episodes": rows + [{"episode_index": 1000}]}, 4,
                      selection_mode="explicit")


def test_external_model_preflight_checks_every_endpoint(monkeypatch):
    seen = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def connect(address, timeout):
        seen.append((address, timeout))
        return Connection()

    monkeypatch.setattr("ablations.preflight.socket.create_connection", connect)
    check_model_endpoints({"model_endpoints": ["http://127.0.0.1:8010/v1", "https://example.test/v1"]})
    assert seen == [(("127.0.0.1", 8010), 5), (("example.test", 443), 5)]
    with pytest.raises(ValueError):
        check_model_endpoints({"model_endpoints": []})


def v4_result(*, nav_success, budget=30.0):
    return {
        "episode_index": 2001,
        "nav_success": nav_success,
        "success": nav_success,
        "interaction_requirement": "required",
        "required_interaction_success": False,
        "required_interaction_completion_fraction": 0.5,
        "reference_path_length_m": 1.0,
        "navigation_path_length_m": 2.0,
        "spl": 0.0,  # The paper report must recompute this value.
        "interaction_precision_episode": 0.5,
        "episode_total_cost": 0.2 if nav_success else 1.0,
        "paper_metric_schema_version": "interactive_nav_v3_paper_metrics_v4",
        "paper_metric_config": {"cost_budget": budget},
        "interaction_action_count": 2,
        "scoring_eligible": True,
        "terminal_reason": "target_found" if nav_success else "policy_exploration_stalled",
    }


def write_episode(root, variant, result):
    task = root / variant / "episode_2001"
    task.mkdir(parents=True)
    result_path = task / "episode_result.json"
    result_path.write_text(json.dumps({"result": result}))
    (task / "batch_task_summary.json").write_text(json.dumps({
        "episode_index": 2001, "completed": True, "runner_exit_code": 0,
        "episode_result_path": str(result_path),
    }))


def test_report_uses_v4_fraction_and_recomputed_spl(tmp_path):
    manifest = {"config": {"episode_indices": [2001]},
                "jobs": [["full", 2001], ["no_outcome_update", 2001]]}
    (tmp_path / "pool_manifest.json").write_text(json.dumps(manifest))
    write_episode(tmp_path, "full", v4_result(nav_success=True))
    write_episode(tmp_path, "no_outcome_update", v4_result(nav_success=False))

    report = summarize(tmp_path)

    assert report["complete"] is True
    assert report["groups"]["full"]["paper_spl"] == 0.5
    assert report["groups"]["no_outcome_update"]["paper_spl"] == 0.0
    assert report["groups"]["full"]["paper_isr"] == 0.5
    assert report["groups"]["no_outcome_update"]["paper_total_cost"] == 1.0
    markdown = (tmp_path / "comparison.md").read_text()
    assert "| Full | 1/1 | 100.0% | 0.500 | 50.0% | 50.0% | 0.20 |" in markdown
    assert "historical" not in markdown.lower()


@pytest.mark.parametrize("change", ["old_schema", "different_budget", "missing_isr"])
def test_report_rejects_incompatible_paper_comparison(tmp_path, change):
    manifest = {"config": {"episode_indices": [2001]},
                "jobs": [["full", 2001], ["no_outcome_update", 2001]]}
    (tmp_path / "pool_manifest.json").write_text(json.dumps(manifest))
    write_episode(tmp_path, "full", v4_result(nav_success=True))
    ablation = v4_result(nav_success=False)
    if change == "old_schema":
        ablation["paper_metric_schema_version"] = "interactive_nav_v3_paper_metrics_v1"
    elif change == "different_budget":
        ablation["paper_metric_config"]["cost_budget"] = 20.0
    else:
        ablation.pop("required_interaction_completion_fraction")
    write_episode(tmp_path, "no_outcome_update", ablation)
    assert summarize(tmp_path)["complete"] is False
    if change == "old_schema":
        markdown = (tmp_path / "comparison.md").read_text()
        assert "| Perception-only Update | 1/1 | — | — | — | — | — |" in markdown


def test_pool_dry_run_renders_all_variants_and_records_full(tmp_path, monkeypatch, capsys):
    source = tmp_path / "selection.json"
    source.write_text(json.dumps({"episodes": [{"episode_index": 2000}]}))
    config = Path(__file__).resolve().parents[2] / "configs/evaluation/ablation_v2_mixed_0_59_20w_dynamic2000_no_recording.json"
    output = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["run_ablation_pool.py", "--config", str(config),
        "--selection-source", str(source), "--output-dir", str(output),
        "--scenes", "1", "--variants", "full", "no_interaction_graph",
        "no_task_decision", "no_outcome_update", "--selection-mode", "stride", "--dry-run"])
    assert main() == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["full_launched"] is True
    assert manifest["config"]["paper_cost_budget"] == 30.0
    assert len(manifest["jobs"]) == 4
    assert not output.exists()
