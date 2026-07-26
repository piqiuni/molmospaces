#!/usr/bin/env python3
"""Validate v3 examples against the schema and cross-field invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_cross_field_invariants(episode: dict[str, Any]) -> None:
    task = episode["task"]
    interactive_nav = episode["interactive_nav"]
    target = interactive_nav["target"]
    interactions = {
        interaction["interaction_id"]: interaction
        for interaction in interactive_nav["interactions"]
    }
    oracle_plans = interactive_nav["oracle_plans"]

    assert task["selection_mode"] == target["selection_mode"]
    assert (
        interactive_nav["success_criteria"]["target_selection"]
        == task["selection_mode"]
    )
    assert (
        interactive_nav["success_criteria"]["distance"]["threshold_m"]
        == task["succ_pos_threshold"]
    )

    if task["selection_mode"] == "specific_instance":
        assert target["selected_instance"] == task["pickup_obj_name"]
        assert task["pickup_obj_candidates"] == [target["selected_instance"]]

    assert interactive_nav["oracle_plan"] == oracle_plans[0]
    plan_ids = [plan["plan_id"] for plan in oracle_plans]
    assert len(plan_ids) == len(set(plan_ids))

    interaction_requirement = interactive_nav["interaction_requirement"]
    if interaction_requirement == "required":
        assert interactions
        assert all(plan["required_interaction_ids"] for plan in oracle_plans)
    elif interaction_requirement == "unnecessary":
        assert not interactions
        assert all(not plan["required_interaction_ids"] for plan in oracle_plans)
        assert all(
            step["type"] != "open_joint"
            for plan in oracle_plans
            for step in plan["steps"]
        )

    initial_interaction_ids = {
        state["interaction_id"]
        for state in interactive_nav["initial_state"]["interaction_states"]
    }
    assert initial_interaction_ids == set(interactions)

    articulation_states = {
        state["joint_name"]: state
        for state in episode["scene_modifications"]["articulation_states"]
    }

    for interaction in interactions.values():
        assert interaction["joint_name"] in articulation_states
        articulation_state = articulation_states[interaction["joint_name"]]
        assert articulation_state["object_name"] == interaction["object_name"]
        assert articulation_state["joint_index"] == interaction["joint_index"]
        assert (
            articulation_state["open_fraction"]
            == interaction["initial_state"]["joint_fraction"]
        )
        for prerequisite in interaction["prerequisites"]:
            prerequisite_id = prerequisite["interaction_id"]
            assert prerequisite_id in interactions
            assert prerequisite_id != interaction["interaction_id"]

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(interaction_id: str) -> None:
        if interaction_id in visiting:
            raise AssertionError(
                f"cyclic interaction dependency at {interaction_id}"
            )
        if interaction_id in visited:
            return
        visiting.add(interaction_id)
        for prerequisite in interactions[interaction_id]["prerequisites"]:
            visit(prerequisite["interaction_id"])
        visiting.remove(interaction_id)
        visited.add(interaction_id)

    for interaction_id in interactions:
        visit(interaction_id)

    for plan in oracle_plans:
        required_interaction_ids = plan["required_interaction_ids"]
        assert all(interaction_id in interactions for interaction_id in required_interaction_ids)
        opened_interaction_ids: list[str] = []
        for step in plan["steps"]:
            interaction_id = step.get("interaction_id")
            if interaction_id is not None:
                assert interaction_id in interactions
            if step["type"] != "open_joint":
                continue
            interaction = interactions[interaction_id]
            assert step["object_name"] == interaction["object_name"]
            assert step["joint_name"] == interaction["joint_name"]
            assert step["joint_index"] == interaction["joint_index"]
            if interaction_id not in opened_interaction_ids:
                opened_interaction_ids.append(interaction_id)
        assert opened_interaction_ids == required_interaction_ids

    plan_by_id = {plan["plan_id"]: plan for plan in oracle_plans}
    for prefix in interactive_nav["generation_validation"]["oracle_prefixes"]:
        plan_id = prefix.get("plan_id")
        if plan_id is not None:
            assert plan_id in plan_by_id
            assert prefix["completed_step_count"] <= len(plan_by_id[plan_id]["steps"])
        for interaction_id in prefix.get("opened_interaction_ids", []):
            assert interaction_id in interactions


def main() -> None:
    schema = _load_json(ROOT / "interactive_nav_episode.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    example_paths = sorted((ROOT / "examples").glob("*.json"))
    if not example_paths:
        raise RuntimeError("No v3 examples found")

    for example_path in example_paths:
        episode = _load_json(example_path)
        validator.validate(episode)
        _validate_cross_field_invariants(episode)
        print(f"PASS {example_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
