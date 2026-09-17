import json
import threading
import time

from ablations.report_pool import summarize
from run_ablation_pool import drain_pool, select_scenes


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


def test_report_uses_manifest_full_rows_and_selected_variant_set(tmp_path):
    full = {
        "episode_index": 2001, "completed": True, "scoring_eligible": True,
        "nav_success": True, "success": True, "task_success": True,
        "required_interaction_success": True, "spl": .5,
        "interaction_precision_episode": 1., "episode_total_cost": 3.,
        "navigation_path_length_m": 2., "interaction_action_count": 1,
        "valid_interaction_attempt_count": 1, "error_interaction_attempt_count": 0,
        "repeated_interaction_attempt_count": 0, "task_irrelevant_interaction_attempt_count": 0,
        "failed_interaction_attempt_count": 0, "step_count": 10,
        "episode_step_budget": 2000, "elapsed_seconds": 4., "terminal_reason": "target_found",
    }
    manifest = {
        "config": {"episode_indices": [2001], "workers": 3, "max_steps": 2000,
                   "step_budget_mode": "dynamic", "recording": False},
        "jobs": [["no_outcome_update", 2001]], "selected_full_rows": [full],
        "m1_refresh_profile": "continuous",
    }
    (tmp_path / "pool_manifest.json").write_text(json.dumps(manifest))
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"result": {**full, "nav_success": False, "success": False}}))
    task = tmp_path / "no_outcome_update/episode_2001"
    task.mkdir(parents=True)
    (task / "batch_task_summary.json").write_text(json.dumps({
        "episode_index": 2001, "completed": True, "runner_exit_code": 0,
        "episode_result_path": str(result_path),
    }))

    report = summarize(tmp_path)

    assert report["complete"] is True
    assert report["groups"]["historical_full"]["reported"] == 1
    assert report["groups"]["no_outcome_update"]["reported"] == 1
    assert report["paired_common_success"]["no_outcome_update"]["indices"] == []
    markdown = (tmp_path / "comparison.md").read_text()
    assert "1 项已报告" in markdown
    assert "3 个共享 worker" in markdown
    assert "Flat Object Memory" not in markdown
