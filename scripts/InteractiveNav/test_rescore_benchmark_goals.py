import copy
import json

import pytest

from scripts.InteractiveNav.evaluation.goal_equivalence import equivalent_target, rescore_result
from scripts.InteractiveNav.rescore_benchmark_goals import aggregate, door_alias, reconstruct_v17_registry, run, sha256


@pytest.fixture
def context():
    return dict(
        target={"selected_instance": "egg1", "category": "egg", "container_name": "fridge",
                "container_aabb_center": [0, 0, 1], "container_aabb_size": [1, 1, 2]},
        candidate_name="egg2",
        objects={"egg1": {"category": "Egg", "parent": "fridge", "room_id": 2},
                 "egg2": {"category": "Egg", "parent": "fridge", "room_id": 2},
                 "fridge": {"category": "Fridge", "parent": "", "room_id": 2}},
        positions={"egg1": [0, 0, 1], "egg2": [.35, 0, .4]},
    )


def test_same_container_accepts_other_egg(context):
    context["target"]["identity_contract"] = {
        "contract_candidates": ["egg1", "egg2"],
    }
    decision = equivalent_target(**context)
    assert decision["reason"] == "same_container"
    assert decision["category_goal_accepted"]
    assert decision["interaction_contract_accepted"]


def test_legacy_same_container_without_contract_proof_is_category_only(context):
    decision = equivalent_target(**context)
    assert decision["category_goal_accepted"]
    assert not decision["interaction_contract_accepted"]
    assert decision["interaction_contract_reason"] == "same_container_without_candidate_contract_proof"


def test_near_cd_on_top_is_not_misclassified_inside(context):
    # Model the near-CD case as two surface objects.  A candidate explicitly
    # declared inside the target container is governed by the frozen AABB
    # check below and must not be rescued by proximity when it lies on top.
    context["target"]["container_name"] = None
    context["objects"]["egg1"]["parent"] = "table"
    context["objects"]["egg2"]["parent"] = "table"
    context["objects"]["table"] = {"category": "Table", "parent": "", "room_id": 2}
    context["positions"]["egg1"] = [0, 0, 1.8]
    context["positions"]["egg2"] = [.275, 0, 1.8]
    decision = equivalent_target(**context)
    assert decision["accepted"]
    assert decision["reason"] == "near_same_category"
    assert not decision["interaction_contract_accepted"]
    assert decision["interaction_contract_reason"] == "proximity_does_not_prove_interaction_contract"


def test_declared_container_candidate_outside_volume_cannot_use_near_fallback(context):
    # Keep XY close enough for the proximity rule but place the candidate above
    # the frozen container volume.  This is inconsistent metadata, not a valid
    # same-category promotion.
    context["positions"]["egg2"] = [0.05, 0.0, 2.1]
    decision = equivalent_target(**context)
    assert not decision["category_goal_accepted"]
    assert decision["reason"] == "not_equivalent"


@pytest.mark.parametrize("change", ["category", "container", "room", "far_top", "unique", "attributes"])
def test_unsafe_substitutions_rejected(context, change):
    if change == "category":
        context["objects"]["egg2"]["category"] = "Tomato"
    elif change == "container":
        context["objects"]["egg2"]["parent"] = "other_fridge"
        context["positions"]["egg2"] = [.2, 0, .4]
    elif change == "room":
        context["objects"]["egg2"]["room_id"] = 3
    elif change == "far_top":
        context["positions"]["egg2"] = [.35, 0, 2]
    else:
        context["target"]["grounding"] = {change: True if change == "unique" else {"color": "red"}}
    assert not equivalent_target(**context)["accepted"]


def test_same_room_category_is_accepted_but_not_contract(context):
    context["target"]["container_name"] = None
    context["objects"]["egg1"]["parent"] = "table"
    context["objects"]["egg2"]["parent"] = "table"
    context["objects"]["table"] = {"category": "Table", "parent": ""}
    context["positions"]["egg2"] = [4, 0, .4]
    decision = equivalent_target(**context)
    assert decision["reason"] == "same_room_category"
    assert decision["category_goal_accepted"]
    assert not decision["interaction_contract_accepted"]
    decision = equivalent_target(**context, same_room=True, door_requirements_equal=False)
    assert decision["category_goal_accepted"]
    assert not decision["interaction_contract_accepted"]
    decision = equivalent_target(**context, same_room=True, door_requirements_equal=True)
    assert decision["accepted"]
    assert not decision["interaction_contract_accepted"]
    assert decision["interaction_contract_reason"] == "shared_channel_does_not_prove_full_contract"


def test_equal_door_is_not_equal_full_contract_without_candidate_plan(context):
    context["target"]["container_name"] = None
    context["objects"]["egg1"]["parent"] = "table"
    context["objects"]["egg2"]["parent"] = "table"
    context["objects"]["table"] = {"category": "Table", "parent": ""}
    context["positions"]["egg2"] = [4, 0, .4]
    door_only = equivalent_target(
        **context, same_room=True, door_requirements_equal=True
    )
    full_contract = equivalent_target(
        **context,
        same_room=True,
        door_requirements_equal=True,
        interaction_requirements_equal=True,
    )
    assert door_only["category_goal_accepted"]
    assert not door_only["interaction_contract_accepted"]
    assert full_contract["interaction_contract_accepted"]


def test_room_rule_cannot_bypass_container(context):
    context["positions"]["egg2"] = [4, 0, .4]
    assert not equivalent_target(**context, same_room=True, door_requirements_equal=True)["accepted"]


def test_same_room_container_target_and_surface_candidate_is_category_only(context):
    context["objects"]["egg2"]["parent"] = "table"
    context["objects"]["table"] = {"category": "Table", "parent": "", "room_id": 2}
    context["positions"]["egg2"] = [4, 0, .4]
    decision = equivalent_target(**context)
    assert decision["reason"] == "same_room_category"
    assert decision["category_goal_accepted"]
    assert not decision["interaction_contract_accepted"]


def test_declared_container_must_match_selected_instance(context):
    context["objects"]["egg1"]["parent"] = "table"
    context["objects"]["table"] = {"category": "Table", "parent": "", "room_id": 2}
    context["positions"]["egg2"] = [4, 0, .4]
    decision = equivalent_target(**context)
    assert not decision["accepted"]
    assert decision["reason"] == "not_equivalent"


@pytest.fixture
def result():
    return dict(status="complete", scoring_eligible=True, nav_success=False, task_success=False,
                success=False, interaction_conditioned_success=False,
                goal_definition_relaxed_success=True, goal_definition_relaxed_reason="verified",
                goal_definition_relaxed_instance_id="obj_000023",
                required_interaction_success=True, sequence_success=True, interaction_requirement="required",
                reference_path_length_m=3, navigation_path_length_m=4, spl=0,
                terminal_reason="target_claim_unverified", step_count=100,
                episode_total_cost=10, episode_total_cost_breakdown={"failure_penalty": 5, "total_cost": 10},
                interaction_precision_episode=.5)


def test_same_container_layers_and_strict_input_immutable(result):
    original = copy.deepcopy(result)
    revised = rescore_result(result, {
        "accepted": True,
        "category_goal_accepted": True,
        "interaction_contract_accepted": True,
        "reason": "same_container",
    })
    assert result == original
    assert not revised["exact_instance_success"]
    assert revised["category_goal_success"]
    assert revised["interaction_contract_goal_success"]
    assert revised["interactive_episode_success"]
    # Layered scoring must not silently rewrite historical strict metrics.
    for key in (
        "nav_success", "task_success", "success", "interaction_conditioned_success",
        "spl", "episode_total_cost", "episode_total_cost_breakdown",
    ):
        assert revised[key] == result[key]
    for key in ("terminal_reason", "step_count", "interaction_precision_episode"):
        assert revised[key] == result[key]


@pytest.mark.parametrize("key,value", [("required_interaction_success", False), ("sequence_success", False)])
def test_same_container_goal_does_not_erase_missing_interaction(result, key, value):
    result[key] = value
    revised = rescore_result(result, {
        "accepted": True,
        "interaction_contract_accepted": True,
    })
    assert revised["category_goal_success"]
    assert revised["interaction_contract_goal_success"]
    assert not revised["interactive_episode_success"]


def test_near_cd_is_category_only_even_when_frozen_plan_completed(result):
    revised = rescore_result(result, {
        "accepted": True,
        "category_goal_accepted": True,
        "interaction_contract_accepted": False,
        "reason": "near_same_category",
    })
    assert revised["category_goal_success"]
    assert not revised["interaction_contract_goal_success"]
    assert not revised["interactive_episode_success"]


@pytest.mark.parametrize("key,value", [
    ("scoring_eligible", False), ("scoring_eligible", None), ("status", "error"),
    ("goal_definition_relaxed_success", False), ("goal_definition_relaxed_reason", "unverified"),
    ("goal_definition_relaxed_instance_id", None),
])
def test_unverified_or_ineligible_cannot_be_promoted(result, key, value):
    result[key] = value
    revised = rescore_result(result, {
        "accepted": True, "interaction_contract_accepted": True,
    })
    assert not revised["category_goal_success"]
    assert not revised["interaction_contract_goal_success"]


def test_unnecessary_interaction_gate_unchanged(result):
    result.update(interaction_requirement="unnecessary", non_interaction_success=False)
    revised = rescore_result(result, {
        "accepted": True, "interaction_contract_accepted": True,
    })
    assert revised["category_goal_success"]
    assert revised["interaction_contract_goal_success"]
    assert not revised["interactive_episode_success"]


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_bad_radius(context, radius):
    with pytest.raises(ValueError):
        equivalent_target(**context, near_radius_m=radius)


def test_registry_requires_recorded_routing_crosscheck(tmp_path):
    xml = tmp_path / "scene.xml"
    xml.write_text('<mujoco><worldbody><body name="doorway_1"><body name="doorway_leaf"><joint type="hinge"/></body></body><body name="egg"/></worldbody></mujoco>')
    metadata = {"objects": {"doorway_1": {"name_map": {"bodies": {"doorway_leaf": "Doorway_door_1"}}}, "egg": {}}}
    episode = {"scene_modifications": {"articulation_states": [{"object_name": "doorway_leaf"}]}}
    log = tmp_path / "eval.log"
    aliases = sorted([door_alias("obj_000001"), door_alias("obj_000002")])
    log.write_text("[v3-interaction-routing] registered_channel_alias_count=2 collision_count=0 aliases=" + ",".join(aliases))
    assert reconstruct_v17_registry(episode, metadata, xml, log)["obj_000003"] == "egg"
    log.write_text("[v3-interaction-routing] registered_channel_alias_count=0 collision_count=0 aliases=")
    with pytest.raises(ValueError, match="routing_alias_mismatch"):
        reconstruct_v17_registry(episode, metadata, xml, log)


def test_double_door_root_is_not_a_skill_alias(tmp_path):
    xml, log = tmp_path / "scene.xml", tmp_path / "eval.log"
    xml.write_text('<mujoco><worldbody><body name="doorway_1"><body name="doorway_a"><joint type="hinge"/></body><body name="doorway_b"><joint type="hinge"/></body></body></worldbody></mujoco>')
    metadata = {"objects": {"doorway_1": {"name_map": {"bodies": {
        "doorway_a": "Double_door_1", "doorway_b": "Double_door_2"}}}}}
    aliases = sorted([door_alias("obj_000002"), door_alias("obj_000003")])
    log.write_text("[v3-interaction-routing] registered_channel_alias_count=2 collision_count=0 aliases=" + ",".join(aliases))
    registry = reconstruct_v17_registry({}, metadata, xml, log)
    assert registry["obj_000001"] == "doorway_1"


def test_aggregate_uses_all_episodes_and_preserves_denominator(result):
    new = rescore_result(result, {
        "accepted": True, "interaction_contract_accepted": True,
    })
    metrics = aggregate([result, new])
    assert metrics["episodes"] == 2
    assert metrics["nav_sr"] == 0
    assert metrics["category_goal_sr"] == .5
    assert metrics["interactive_episode_sr"] == .5
    assert metrics["mean_total_cost"] == 10


def test_aggregate_falls_back_to_frozen_strict_fields_for_legacy_rows(result):
    legacy_success = copy.deepcopy(result)
    legacy_success.update(
        nav_success=True,
        success=True,
        interaction_conditioned_success=True,
    )
    # Simulate a pre-layer result: no newly introduced fields are present.
    for key in (
        "exact_instance_success",
        "category_goal_success",
        "interaction_contract_goal_success",
        "interactive_episode_success",
    ):
        legacy_success.pop(key, None)
    metrics = aggregate([legacy_success])
    assert metrics["exact_instance_sr"] == 1.0
    assert metrics["category_goal_sr"] == 1.0
    assert metrics["interaction_contract_goal_sr"] == 1.0
    assert metrics["interactive_episode_sr"] == 1.0


def test_output_overwrite_rejected(tmp_path):
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        run(tmp_path, tmp_path / "benchmark.json", tmp_path, tmp_path, .3)


def test_end_to_end_versioned_report_preserves_inputs(tmp_path, context, result):
    evaluation = tmp_path / "evaluation"
    attempt = evaluation / "episode_0000/attempt_001"
    result_path = attempt / "eval/episodes/0000_case/episode_result.json"
    result_path.parent.mkdir(parents=True)
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    benchmark = tmp_path / "benchmark.json"
    episode = {"house_index": 1, "data_split": "val", "interactive_nav": {
        "case_id": "case", "target": context["target"]}, "scene_modifications": {
        "object_poses": context["positions"], "articulation_states": []}}
    episode["interactive_nav"]["target"]["identity_contract"] = {
        "contract_candidates": ["egg1", "egg2"],
    }
    benchmark.write_text(json.dumps([episode]))
    (scenes / "val_1_metadata.json").write_text(json.dumps({"objects": context["objects"]}))
    (scenes / "val_1.xml").write_text('<mujoco><worldbody><body name="egg1"/><body name="egg2"/><body name="fridge"/></worldbody></mujoco>')
    (attempt / "eval.log").write_text("[v3-interaction-routing] registered_channel_alias_count=0 collision_count=0 aliases=")
    (attempt / "eval/run_manifest.json").write_text(json.dumps({
        "benchmark_sha256": sha256(benchmark), "protocol_version": "interactive_nav_v3_benchmark_eval_v17"}))
    result.update(episode_index=0, case_id="case", domains=["container"], goal_definition_relaxed_instance_id="obj_000002")
    result_path.write_text(json.dumps({"result": result}))
    result_path.with_name("episode_visualization.json").write_text(json.dumps({"case_id": "case", "target": {}}))
    summary_path = evaluation / "summary.json"
    summary_path.write_text(json.dumps({"episodes": [{"episode_index": 0,
        "episode_result_path": str(result_path), "attempt_dir": str(attempt)}]}))
    original_hashes = {p: sha256(p) for p in (summary_path, result_path, benchmark)}
    output = tmp_path / "rescore"
    report = run(evaluation, benchmark, scenes, output, .3)
    assert report["promoted_episode_indices"] == [0]
    assert report["interaction_contract_promoted_episode_indices"] == [0]
    assert report["aggregates"]["all"]["revised_all"]["category_goal_sr"] == 1
    assert report["aggregates"]["all"]["revised_all"]["interactive_episode_sr"] == 1
    assert report["aggregates"]["all"]["revised_all"]["nav_sr"] == 0
    assert report["aggregates"]["all"]["strict_all"]["nav_sr"] == 0
    rescored = json.loads((output / "rescored_results.json").read_text())[0]
    assert not rescored["success"]
    assert rescored["category_goal_success"]
    assert rescored["interactive_episode_success"]
    assert all(sha256(p) == digest for p, digest in original_hashes.items())
