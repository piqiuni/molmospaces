import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).with_name("run_rule_mllm_image_qa.py")
SPEC = importlib.util.spec_from_file_location("run_rule_mllm_image_qa", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def sample_module2_request() -> dict:
    return {
        "schema_version": 4,
        "instruction": "Rank the concrete subgoals. Return compact JSON.",
        "mission": {"mode": "explore_all"},
        "robot": {"current_room": "room_1"},
        "recent_decisions": [],
        "graph": {"rooms": [{"id": "room_1"}], "portals": [], "containers": []},
        "room_object_reasoning": {"stage": "observed_room_target_plausibility"},
        "candidates": [{"id": "frontier:1", "action": "explore", "distance_m": 1.0}],
    }


def test_module2_cases_are_text_only_and_do_not_materialize_a_frame(tmp_path):
    cases = qa.materialize_module2_cases(
        tmp_path,
        [
            {
                "module2_request": sample_module2_request(),
                "selection_step": 42,
                "candidate_sequence": 7,
                "heldout": {"rule_reference_selected_candidate_id": "frontier:1"},
            }
        ],
    )

    assert len(cases) == 1
    case = cases[0]
    assert case["stage"] == qa.MODULE2_STAGE
    assert case["modality"] == "text"
    assert "image_path" not in case
    assert "raw_full_frame_path" not in case
    assert "boxes" not in case["public_context"]
    assert "/no_think" in case["prompt"]
    assert not (tmp_path / "annotated_frames").exists()


def test_module2_model_call_uses_only_text_content():
    class Completions:
        def __init__(self):
            self.kwargs = {}

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ranked_ids":["frontier:1"]}'))],
                usage=None,
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    case = qa.materialize_module2_cases(
        Path("/unused"),
        [{"module2_request": sample_module2_request(), "heldout": {}}],
    )[0]

    result = qa.call_model(
        case,
        SimpleNamespace(model="local", max_tokens=32, temperature=0.0),
        client,
    )

    assert result["parsed_response"] == {"ranked_ids": ["frontier:1"]}
    assert completions.kwargs["messages"][0] == {
        "role": "system",
        "content": "Return only a valid JSON object.",
    }
    assert completions.kwargs["messages"][1]["content"] == [
        {"type": "text", "text": case["prompt"]}
    ]
    assert "image_url" not in str(completions.kwargs["messages"])
    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_html_report_renders_module2_without_an_image(tmp_path):
    module2_case = qa.materialize_module2_cases(
        tmp_path, [{"module2_request": sample_module2_request(), "heldout": {}}]
    )[0]
    report = qa.render_html_report(
        tmp_path,
        [module2_case],
        [
            {
                "case_id": module2_case["case_id"],
                "stage": qa.MODULE2_STAGE,
                "parsed_response": {"ranked_ids": ["frontier:1"]},
            }
        ],
    )

    page = report.read_text(encoding="utf-8")

    assert "No image is sent to M2." in page
    assert "module2_candidate_selection_001" in page
    assert "<img" not in page


def test_module2_request_uses_the_production_request_builder():
    snapshot = {
        "robot_xy": [0.0, 0.0],
        "exploration_context": {"proposal_count": 1},
        "target_context": {"enabled": False},
        "graph_context": {"nodes": [], "edges": []},
        "candidates": [
            {
                "candidate_id": "frontier:1",
                "behavior_type": "EXPLORE",
                "source": "test",
                "target_id": "frontier:1",
                "target_name": "Frontier",
                "goal_xyyaw": [1.0, 0.0, 0.0],
                "features": {"distance_m": 1.0},
                "metadata": {},
            }
        ],
    }
    trace = {
        "candidate_curation": {
            "selected_ids": ["frontier:1"],
            "quality_by_id": {"frontier:1": 1.0},
            "quality_terms_by_id": {"frontier:1": {"information_gain": 1.0}},
            "decision_hint_by_id": {"frontier:1": "INFORMATION_GAIN"},
            "entered_room_ids": [],
        },
        "recent_decisions": [],
    }

    request, provenance = qa.build_module2_request(snapshot, trace)

    assert request["schema_version"] == 4
    assert request["candidates"][0]["id"] == "frontier:1"
    assert "Rank the concrete subgoals" in request["instruction"]
    assert provenance["request_builder"].endswith("ModelPolicyClient.build_request")
