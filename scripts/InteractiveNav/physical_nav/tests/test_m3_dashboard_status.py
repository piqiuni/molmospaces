"""Exercise the browser card without ROS, model calls, or robot commands."""
import json
import shutil
import subprocess

import pytest

from physical_six_panel_server import _HTML


def render(payload):
    if not shutil.which("node"):
        pytest.skip("node unavailable")
    function = _HTML.split("function renderM3(m3){", 1)[1].split("\nfunction ", 1)[0]
    script = """
const labels=[];
const make=(tag,cls,value)=>({append(){},replaceChildren(){},value:labels.push(String(value??''))});
const q=()=>make('div');
const firstObj=(...xs)=>xs.find(x=>x&&typeof x==='object'&&Object.keys(x).length)||{};
const text=x=>String(x??'');const classZh=text;const clip=text;
""" + "function renderM3(m3){" + function
    script += "\nrenderM3(" + json.dumps(payload) + ");console.log(JSON.stringify(labels));"
    return json.loads(subprocess.check_output(["node", "-e", script], text=True))


def test_approach_never_displays_started_or_starts_countdown():
    labels = render({"execution_state": {"state": "APPROACH_INTERACTION"},
                     "behavior_feedback": {"status": "STARTED"}})
    assert "APPROACHING" in labels
    assert "STARTED" not in labels
    assert not any("策略剩余" in label for label in labels)


def test_room_m1_ui_shows_text_input_real_prompt_and_evidence():
    if not shutil.which("node"):
        pytest.skip("node unavailable")
    names = ("isRoomM1", "m1DisplayName", "m1Prompt", "m1Answer")
    functions = [line for line in _HTML.splitlines()
                 if any(line.startswith("function " + name + "(") for name in names)]
    payload = {"instruction": "Actual room instruction", "context": {
        "objects": [{"object_id": "chair1", "category": "chair", "currently_visible": False}]},
        "raw_text": json.dumps({"room_id": 1, "room_attribute": "unknown", "confidence": .2,
                                "evidence_object_ids": ["chair1"]})}
    script = "const text=x=>String(x??'');\n" + "\n".join(functions)
    script += "\nconst e=" + json.dumps(payload) + ";console.log(JSON.stringify([m1DisplayName(e),m1Prompt(e),m1Answer(e)]));"
    label, prompt, answer = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert "纯文本" in label and "无图像" in label
    assert "Actual room instruction" in prompt and "chair1: chair" in prompt
    assert "历史观测" in prompt
    assert "evidence_object_ids" in answer and "chair1: chair" in answer
    assert "模型自报" in answer


def test_m2_idle_displays_terminal_mission_reason():
    if not shutil.which("node"):
        pytest.skip("node unavailable")
    function = _HTML.split("function m2EventsWithLiveState(s){", 1)[1].split("\nconst eventSignatures", 1)[0]
    script = 'const text=x=>String(x??" ");\nfunction m2EventsWithLiveState(s){' + function
    script += '\nconsole.log(JSON.stringify(m2EventsWithLiveState({navigation:{goal_status:{status:"EXPLORATION_STALLED",timestamp:2,detail:{reason:"semantic_mission_no_progress"}}}})));'
    events = json.loads(subprocess.check_output(["node", "-e", script], text=True))
    assert "EXPLORATION_STALLED" in events[-1]["raw_text"]


def test_running_policy_has_countdown_and_terminal_timeout_wins():
    payload = {"execution_state": {"state": "INTERACTING", "decision_id": "d"},
               "policy_event": {"stage": "STARTED", "decision_id": "d", "policy_deadline_at": 9999999999}}
    assert any("策略剩余" in label for label in render(payload))
    payload["policy_event"].update(stage="FINISHED", timestamp=10,
                                   result={"status": "TIMEOUT", "success": False})
    labels = render(payload)
    assert "TIMEOUT" in labels
    assert not any("策略剩余" in label for label in labels)
