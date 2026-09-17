import copy

import pytest

from scripts.InteractiveNav.audit_mixed_benchmark_run import classify_episode, scan_log, score_profiles, run


@pytest.fixture
def episode():
    return {"interactive_nav": {"interaction_domains": ["channel", "container"],
        "interaction_requirement": "required", "initial_state": {"container_joints_closed": True},
        "target": {"selected_instance": "pencil"},
        "interactions": [{"interaction_id": "drawer", "type": "container_sliding_drawer",
                          "effect_types": ["reveal_target_object"]}],
        "oracle_plan": {"required_interaction_ids": ["door", "drawer"]}}}


@pytest.fixture
def result():
    return {"status": "complete", "scoring_eligible": True, "policy_name": "ros_object_goal_rule",
            "nav_success": True, "terminal_reason": "target_found", "required_interaction_success": False,
            "interaction_attempts": [{"command_id": "door", "node_type": "portal", "success": True}]}


def test_strict_success_is_not_exempt_from_scene_audit(episode, result):
    original = copy.deepcopy(result)
    audit = classify_episode(episode, result, [], {})
    assert audit["scene_exclusion_reasons"] == ["strict_target_reached_without_required_container_interaction"]
    assert result == original


@pytest.mark.parametrize("change", ["category_only", "container_attempt", "unknown_attempt", "not_closed", "not_required"])
def test_counterexample_requires_complete_evidence(episode, result, change):
    if change == "category_only":
        result.update(nav_success=False, goal_definition_relaxed_success=True)
    elif change == "container_attempt":
        result["interaction_attempts"].append({"node_type": "container", "success": True})
    elif change == "unknown_attempt":
        result["interaction_attempts"].append({"success": True})
    elif change == "not_closed":
        episode["interactive_nav"]["initial_state"]["container_joints_closed"] = False
    else:
        episode["interactive_nav"]["oracle_plan"]["required_interaction_ids"] = ["door"]
    assert not classify_episode(episode, result, [], {})["scene_exclusion_reasons"]


def test_existing_runtime_invalid_reasons_preserved(episode, result):
    result.update(scoring_eligible=False, nav_success=False,
                  scoring_exclusion_reasons=["initial_target_visibility_mismatch"], step_count=0)
    audit = classify_episode(episode, result, [], {})
    assert audit["scene_exclusion_reasons"] == ["initial_target_visibility_mismatch"]


@pytest.fixture
def scan_result():
    return {"result": {"command_id": "scan", "sequence_type": "drawer_scan",
                       "failure_reason": "drawer_view_restore_timeout", "success": False,
                       "region_results": [{"success": True}, {"success": True}],
                       "final_close_success": True,
                       "view_restore_convergence": {"checked": True, "converged": False}}}


def test_restore_fault_is_execution_quarantine_not_scene_invalid(episode, result, scan_result):
    result.update(nav_success=False, interaction_attempts=[{
        "command_id": "scan", "success": False, "failure_reason": "drawer_scan_execution_failed"}])
    audit = classify_episode(episode, result, [("scan.json", scan_result)], {})
    assert not audit["scene_exclusion_reasons"]
    assert audit["execution_exclusion_reasons"] == ["drawer_view_restore_timeout_after_successful_scan"]


@pytest.mark.parametrize("change", ["wrong_command", "physical_failure", "close_failed", "no_convergence_check"])
def test_scan_failure_alone_not_proof_of_restore_fault(episode, result, scan_result, change):
    result.update(nav_success=False, interaction_attempts=[{
        "command_id": "scan", "success": False, "failure_reason": "drawer_scan_execution_failed"}])
    if change == "wrong_command":
        scan_result["result"]["command_id"] = "unrelated"
    elif change == "physical_failure":
        scan_result["result"]["region_results"][0]["success"] = False
    elif change == "close_failed":
        scan_result["result"]["final_close_success"] = False
    else:
        scan_result["result"].pop("view_restore_convergence")
    assert not classify_episode(episode, result, [("scan.json", scan_result)], {})["execution_exclusion_reasons"]


def test_private_discovery_not_sufficient_to_exclude_or_promote(episode, result):
    result["nav_success"] = False
    audit = classify_episode(episode, result, [], {"transient_target_discovery": {"visible_pixels": 326}})
    assert audit["review_flags"]
    assert not audit["scene_exclusion_reasons"] and not audit["execution_exclusion_reasons"]
    assert not result["nav_success"]


def test_shutdown_errors_separated_from_active_errors(tmp_path):
    log = tmp_path / "roslaunch.log"
    log.write_text("move_base must be in an inactive state to make a plan\n"
                   "[mapping] killing on exit\nTraceback (most recent call last):\n"
                   "publish() to a closed topic\ndouble free or corruption (out)\n")
    summary = scan_log(log)
    assert summary["active_counts"] == {"active_make_plan_rejected": 1}
    assert summary["shutdown_counts"]["traceback"] == 1
    assert summary["examples"]["shutdown:closed_topic"]["line"] == 4


def test_profiles_remove_success_and_failure_without_label_changes():
    rows = [{"episode_index": i, "domains": ["channel", "container"], "scoring_eligible": i != 0,
             "nav_success": i in (1, 3), "success": i == 3} for i in range(4)]
    original = copy.deepcopy(rows)
    audits = [{"episode_index": i, "scene_exclusion_reasons": ["scene"] if i in (0, 1) else [],
               "execution_exclusion_reasons": ["restore"] if i == 2 else []} for i in range(4)]
    scores, retained = score_profiles(rows, audits)
    assert scores["all_planned"]["mixed"]["episodes"] == 4
    assert scores["original_eligible"]["mixed"]["episodes"] == 3
    assert scores["scene_valid"]["mixed"]["episodes"] == 2
    assert scores["scene_and_execution_valid"]["mixed"]["episodes"] == 1
    assert [row["episode_index"] for row in retained] == [3]
    assert scores["scene_valid"]["mixed"]["nav_successes"] == 1
    assert original == rows


def test_output_overwrite_rejected(tmp_path):
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(tmp_path, tmp_path, tmp_path)
