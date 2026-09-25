import json

from ablations.progress import format_pool


def test_progress_shows_completed_v4_means_and_separates_run_errors(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "pool_manifest.json").write_text(json.dumps({
        "jobs": [["no_task_decision", 2001], ["no_task_decision", 2002],
                 ["no_task_decision", 2003]],
    }))
    (pool / "pool_status.json").write_text(json.dumps({
        "elapsed_sec": 90, "active": {"0": ["no_task_decision", 2003]},
        "per_variant": {"no_task_decision": {"reported": 2}},
    }))
    completed = pool / "no_task_decision/episode_2001"
    completed.mkdir(parents=True)
    result = completed / "episode_result.json"
    result.write_text(json.dumps({"result": {
        "scoring_eligible": True,
        "paper_metric_schema_version": "interactive_nav_v3_paper_metrics_v4",
        "task_success": True, "interaction_conditioned_success": True,
        "nav_success": True, "spl": 0.5,
        "required_interaction_completion_fraction": 1.0,
        "interaction_precision_episode": 0.25, "episode_total_cost": 0.4,
        "step_count": 300, "elapsed_seconds": 120.0,
    }}))
    (completed / "batch_task_summary.json").write_text(json.dumps({
        "completed": True, "episode_result_path": str(result),
    }))
    failed = pool / "no_task_decision/episode_2002"
    failed.mkdir(parents=True)
    (failed / "batch_task_summary.json").write_text(json.dumps({"completed": False}))

    view = format_pool(pool)
    assert "Greedy 1.5min" in view
    assert "完成 1/3 | 运行/收尾 1 | 排队 0 | 运行异常 1" in view
    assert "已完成均值（n=1）：TaskSR=1.000 | ICS=1.000 | NavSR=1.000" in view
    assert "ISR=1.000 | SPL=0.500 | IP=0.250 | Cost=0.400" in view
    assert "Steps=300.0 | Eval=120.0s" in view


def test_progress_waits_for_manifest(tmp_path):
    assert "等待任务清单" in format_pool(tmp_path)
