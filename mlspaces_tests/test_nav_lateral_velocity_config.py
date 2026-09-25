import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dwa_lateral_braking_covers_the_allowed_speed_in_one_control_period():
    path = ROOT / "Interactive-Nav-SG-nav/src/nav_pkg/configs/controller/dwa_controller_params.yaml"
    config = yaml.safe_load(path.read_text())["DWAPlannerROS"]
    assert config["min_vel_y"] == config["max_vel_y"] == 0.0
    assert config["vy_samples"] == 1
    assert config["acc_lim_y"] / config["controller_frequency"] >= config["max_vel_trans"]


def test_sim_navigation_disables_lateral_commands_and_passes_the_flag_to_bridge():
    source = ROOT / "scripts/InteractiveNav/run_nav_ros_sim.py"
    tree = ast.parse(source.read_text())
    arguments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--allow_lateral_cmd_vel"
    ]
    assert len(arguments) == 1
    default = next(keyword.value for keyword in arguments[0].keywords if keyword.arg == "default")
    assert ast.literal_eval(default) is False
    bridge = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "RosBridgePolicy")
    flag = next(keyword.value for keyword in bridge.keywords if keyword.arg == "allow_lateral_cmd_vel")
    assert isinstance(flag, ast.Attribute) and flag.attr == "allow_lateral_cmd_vel"
    assert isinstance(flag.value, ast.Name) and flag.value.id == "args"
